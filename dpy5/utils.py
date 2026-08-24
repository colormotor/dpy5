r"""
 _____              _____ 
|  __ \            | ____|
| |  | |_ __  _   _| |__  
| |  | | '_ \| | | |___ \ 
| |__| | |_) | |_| |___) |
|_____/| .__/ \__, |____/ 
       | |     __/ |      
       |_|    |___/

Processing-like API for DiffVG
© Daniel Berio (@colormotor) 2026 - ...

Various helpers
"""

from collections import defaultdict
import torch
import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
from . import diff_canvas
from PIL import Image


class CanvasOptimizer:
    """Helper for optimization loops. Wraps a `DiffCanvas` instance
    and provides entry points for:
    - initialization (`init`, optional)
    - setting up optimizers (`setup`)
    - drawing at each iteration (`draw`)
    - loss computation (`loss`)
    - clipping values or other post processing oparations (`postprocess`, optional).

    Subclass `CanvasOptimizer` and override these methods, and then simply call `step` at each iteration:

    E.g.:

    ```python
    class MyCanvasOpt(CanvasOptimizer):
        def init(self, c):
            # Initialize parameters once before draw (optional)
            ...

        def draw(self, c):
            c.background(1.0)
            ....
            return c.render()

        def postprocess(self, c):
            # Clamp values if necessary
            with torch.no_grad():
                for v in c.get_vars('some variables'):
                    v.data.clamp_(0, 1)

        def setup(self, c):
            self.optimizers = [
               torch.optim.Adam(c.get_vars('some variables'), lr=1.0),
            ]
            # optionally add schedulers to `self.schedulers`


        def loss(self, img):
            # Compute some loss based on img (the output of the renderer at each step)
            loss = ...
            return loss

    opt = MyCanvasOpt(w, h)
    ```
    """

    def __init__(self, w, h, num_opt_steps, verbose=False):
        self.c = diff_canvas.DiffCanvas(w, h)
        self.w = w
        self.h = h
        self.optimizers = []
        self.schedulers = []
        self.running = False
        self.epoch = 0
        self.num_opt_steps = num_opt_steps
        self.verbose = verbose
        self.init(self.c)
        self.img = self.draw(self.c)
        if self.img is None:  # If user forgets to return img
            self.img = self.c.img

    ######################################
    # Functions for user to override
    def init(self, c):
        """called once before drawing starts, can be used to initialize variables"""
        pass

    def setup(self, c):
        """Setup optimizers and schedulers
        these must be added to `self.optimizers` and `self.schedulers`.
        The lists will be cleared automatically when calling `run`
        """
        pass

    def release(self):
        """Optionally release heavy-weight objects"""
        pass

    def draw(self, c):
        """Construct/render the scene. Called every step."""
        pass

    def loss(self, img):
        """Compute and return loss for each step"""
        return 0.0

    def postprocess(self, c):
        """Optionally clamp values or other procedures after each opt step"""
        pass

    ######################################
    # Built in

    def step(self):
        # Peform an optimization step if optimizing
        if not self.running:
            return

        for opt in self.optimizers:
            opt.zero_grad()

        img = self.draw(self.c)
        if img is None:  # If user forgets to return img
            img = self.c.img

        self.img = img

        if self.epoch >= self.num_opt_steps:
            print("Stopping", self.epoch)
            self.running = False
            self.release()
            return

        loss = self.loss(img)
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        for sched in self.schedulers:
            sched.step()
        self.postprocess(self.c)
        self.epoch += 1

    def run(self, num_steps=0):

        self.optimizers = []
        self.schedulers = []
        self.epoch = 0
        if num_steps:
            self.num_opt_steps = num_steps
        self.setup(self.c)
        self.running = True

    def get_image(self):
        return Image.fromarray((self.img.detach().cpu().numpy() * 255).astype(np.uint8))


class MultiLoss:
    """Helper for managing multiple losses organized by name and input
    Keeps track of losses and enables visualization with matplotlib or py5canvas (implot)

    Losses are registered with a name, callable, scalar weight, and an optional input
    format list.  Calling the instance evaluates all registered losses and any
    extra tensor kwargs whose name contains ``'loss'``, sums them, and records
    per-term and total values.

    **Input format convention**
    ---------------------------
    The ``inputs`` argument of ``add`` controls how the call arguments are
    assembled:

    * ``None``              -> ``fn(default)``
    * ``[None, 'target']``  -> ``fn(default, kwargs['target'])``
    * ``['a', 'b']``        -> ``fn(kwargs['a'], kwargs['b'])``

    ``None`` inside a list refers to the first positional argument.

    **Example usage**
    -----------------

    Example
    -------
    ```
    ml = MultiLoss()
    ml.add_loss('rgb', F.l1_loss, 1.0, inputs=[None, 'target'])
    ml.add_loss('lpips', lpips_fn, 0.1, inputs=['aux', 'target'])
    ...
    loss = ml(pred_img, target=target_img, aux=aux_img)
    ```
    """

    def __init__(self, max_history=1000):
        self.items = {}  # name -> [loss_fn, weight]
        self.formats = {}  # name -> input format list / None
        self.history = defaultdict(list)
        self.max_history = max_history

    def add(self, name, fn, weight, inputs=None):
        """Register a loss.
        inputs: None          -> fn(default)
                [None, 'b']   -> fn(default, kwargs['b'])
                ['a', 'b']    -> fn(kwargs['a'], kwargs['b'])
        """
        self.items[name] = [fn, weight]
        self.formats[name] = inputs

    def has_loss(self, name):
        return name in self.items and self.items[name][1] > 0.0

    def update_weight(self, name, weight):
        if name in self.items:
            self.items[name][1] = weight

    def set_weights(self, **weights):
        for name, w in weights.items():
            if name in self.items:
                self.items[name][1] = w

    def __call__(self, default=None, **kwargs):
        call_args = {}
        for name, fmt in self.formats.items():
            if fmt is None:
                call_args[name] = (default,)
            else:
                args = []
                for spec in fmt:
                    if spec is None:
                        args.append(default)
                    elif spec in kwargs:
                        args.append(kwargs[spec])
                    else:
                        raise ValueError(
                            f"expected `{spec}` for loss `{name}` but did not get it"
                        )
                call_args[name] = tuple(args)

        # Trick to inject external losses (TODO test)
        for key, val in kwargs.items():
            if "loss" in key and isinstance(val, torch.Tensor):
                # if key not in self.items:
                call_args[key] = val

        total = None
        for name, args in call_args.items():
            if name in self.items:
                fn, w = self.items[name]
                if w <= 0.0:
                    continue
                l = fn(*args) * w
            elif isinstance(args, torch.Tensor):
                l = args
            else:
                continue

            if not isinstance(l, torch.Tensor):
                l = torch.tensor(l, dtype=torch.float32)

            total = l if total is None else total + l

            self.history[name].append(float(l.detach().cpu()))
            if len(self.history[name]) > self.max_history:
                self.history[name] = self.history[name][-self.max_history :]

        if total is None:
            total = torch.tensor(0.0)

        self.history["total"].append(float(total.detach().cpu()))
        return total

    def get_losses(self):
        return {k: list(v) for k, v in self.history.items()}

    def to_string(self, n=1):
        res = ""
        for key in self.items:
            v = self.history.get(key, [])
            if v:
                res += f"{key} w:{self.items[key][1]:.4f} loss:{v[-1]:.6f}\n"
        return res

    def plot(self, n=None, title="Loss"):
        plt.title(title)
        if n is None:
            n = self.max_history
        for key, kloss in self.history.items():
            if key == "total":
                continue
            if not kloss:
                continue
            plt.plot(kloss[-n:], label=f"{key}:{kloss[-1]:.4f}")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("loss")

    def implot(self, n=50):
        """Plot inthe UI of an interactive py5canvas sketch"""
        from slimgui import implot, imgui

        parts = []
        if implot.begin_plot("Loss", size=[-1, 200]):
            implot.setup_axes(
                None, None, implot.AxisFlags.AUTO_FIT, implot.AxisFlags.AUTO_FIT
            )
            for key, kloss in self.history.items():
                if key == "total":
                    continue
                if not self.has_loss(key) and "loss" not in key:
                    continue
                if not kloss:
                    continue
                arr = np.array(kloss[-n:], dtype=np.float32)
                arr = arr - np.mean(arr)
                last_s = f"{float(arr[-1]):.3f}"
                parts.append(f"{key}: {last_s}")
                implot.plot_line(f"{key}##loss_{key}", arr)
            implot.end_plot()

        if parts:
            n_cols = 3
            for i, text in enumerate(parts):
                if i % n_cols != 0:
                    imgui.same_line()
                imgui.text(text)


def show_animation(
    fig, frame, num_frames, ax=None, filename="", fps=30, headless=False
):
    """Show a matplotlib animated plot, each frame is rendered by automatically calling `frame`
    Optionally saves a video, if opencv is installed and `filename` is defined
    """
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    frames = []
    ani = None

    if ax is None:
        ax = plt.gca()

    state = lambda: None
    state.playing = True

    def _frame(i):

        plt.clf()
        try:
            cur_frame = frame(i)
        except Exception as e:
            if ani is not None:
                ani.event_source.stop()
            print(e)
            raise e

        fig = plt.gcf()
        if filename:
            if cur_frame is None:
                buf = BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                image = Image.open(buf)
            else:
                if isinstance(cur_frame, np.ndarray):
                    if cur_frame.dtype != np.uint8:
                        cur_frame = (cur_frame * 255).astype(np.uint8)
                    cur_frame = Image.fromarray(cur_frame)
                image = cur_frame.convert("RGB")
            frames.append(np.array(image)[:, :, :3])
        if i == num_frames - 1:
            print("Animation ended")
            if ani is not None:
                ani.event_source.stop()
            state.playing = False

    norepeat = os.getenv("QUIT_ANIMATION") == "1"

    if not headless:
        ani = FuncAnimation(
            fig,
            _frame,
            frames=num_frames,
            blit=False,
            repeat=not norepeat,
            interval=1000 / fps,
        )

        # Ugly workaround because func anim does not let us close
        if norepeat:
            import threading

            def monitor_animation():
                while state.playing:
                    time.sleep(0.5)
                print("Closing animation")
                plt.close(fig)

            monitor_thread = threading.Thread(target=monitor_animation)
            monitor_thread.start()
        try:
            plt.show()
        except AttributeError as e:
            print("Closing")
    else:
        for i in range(num_frames):
            print("Headless: step %d of %d" % (i + 1, num_frames))
            _frame(i)

    if filename and frames:
        print("saving %d frames" % len(frames))
        import cv2

        fmt = cv2.VideoWriter_fourcc(*"mp4v")
        h, w, _ = frames[0].shape
        video_writer = cv2.VideoWriter(filename, fmt, fps, (w, h))
        for frame in frames:
            video_writer.write(frame[:, :, ::-1])
        video_writer.release()
        print("Finished saving")


class perf_timer:
    def __init__(self, name="", verbose=True):
        # if name and verbose:
        #    print(name)
        self.name = name
        self.verbose = verbose
        self.elapsed = 0

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t) * 1000
        if self.name and self.verbose:
            print("%s: elapsed time %.3f milliseconds" % (self.name, self.elapsed))
