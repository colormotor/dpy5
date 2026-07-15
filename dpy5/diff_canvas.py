"""
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

Core API
"""

import numbers
import torch
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
import pydiffvg
import copy, time
from PIL import Image

use_gpu = torch.cuda.is_available()
print("Using gpu: ", use_gpu)
pydiffvg.set_use_gpu(use_gpu)

npy = lambda v: v.detach().cpu().numpy()


def make_mat(M, device, dtype):
    return torch.vstack(
        [
            torch.stack([torch.as_tensor(v, dtype=dtype, device=device) for v in row])
            for row in M
        ]
    )


def make_vec(*args, device, dtype):
    return torch.stack([torch.as_tensor(v, dtype=dtype, device=device) for v in args])


def is_number(x):
    return isinstance(x, numbers.Number) or (
        isinstance(x, torch.Tensor) and x.numel() == 1
    )


class CanvasState:
    """Keeps track of styles etc to enable push/pop"""

    def __init__(self, c):
        self.c = c
        self.cur_fill = c._get_color(1.0)
        self.cur_stroke = c._get_color(0.0)
        self._rect_mode = "corner"
        self._ellipse_mode = "center"
        self._line_width = 1.0
        self._angle_mode = "radians"
        self._tension = 0.5
        self._fill_rule = "evenodd"

    def set(self, prev=None):
        """Called if pop_style/pop is called in canvas"""

        def should_set(prev, name):
            if prev is None:
                return True
            return prev.__dict__[name] != self.__dict__[name]

        # Call function if necessary
        # if should_set(prev, "_line_width"):
        #     self.c.stroke_weight(self._line_width)

        # if should_set(prev, "_tension"):
        #     self.c.curve_tightness(self._tension)


def draw_states_properties(*names):
    def decorator(cls):
        for name in names:

            def getter(self, n=name):
                return getattr(self.draw_states[-1], n)

            def setter(self, value, n=name):
                setattr(self.draw_states[-1], n, value)

            setattr(cls, name, property(getter, setter))
        return cls

    return decorator


# Style properties, automatically adds setters and getters
@draw_states_properties(
    "cur_fill",
    "cur_stroke",
    "_rect_mode",
    "_ellipse_mode",
    "_line_width",
    "_angle_mode",
    "_fill_rule",
    "_tension",
)
class DiffCanvas:
    """
    A differentiable 2D vector graphics canvas based on DiffVG, using a Processing-like syntax.
    It provides an imperative drawing API similar to Processing/p5
    (e.g. `background`, `fill`, `stroke`, `push`, `pop`, `translate`, `rotate`,
    `scale`, `line`, `rect`, `ellipse`, `begin_shape` / `end_shape`) while
    building a scene using DiffVG.  All geometric and style parameters
    are PyTorch tensors, so gradients flow through `render()` and the resulting
    image can be optimized with gradient descent.
    """

    def __init__(self, width, height, device=None):
        self.vars = defaultdict(list)
        if device is None:
            device = default_device()
        self.device = device
        self.dtype = torch.float32
        self.cur_shape = None

        self._width = width
        self._height = height
        self._bg = None
        self.clear_vars()
        self.reset()

    def reset(self):
        self.items = []
        self.mat_stack = [torch.eye(3, device=self.device, dtype=self.dtype)]

        # Keep track of draw states
        self.draw_states = [CanvasState(self)]
        self.draw_states[-1].set()

        self.building_shape = False

        self.primitives = []
        self.groups = []

        # Cache for shapes that can be instanced
        # Gives corresponding indices in primitive list
        self.shape_to_inds = {}
        self.img = None

        # Reset counter for auto var id
        self._var_counters = defaultdict(int)

    # def begin(self):
    #     @contextmanager
    #     def popmanager():
    #         pass
    #         try:
    #             yield
    #         finally:
    #             self.end()

    #     self.reset()
    #     self.building = True
    #     return popmanager()

    # def end(self):
    #     self.building = False

    def push_matrix(self):
        """
        Save the current transformation
        """

        @contextmanager
        def popmanager():
            pass
            try:
                yield
            finally:
                self.pop_matrix()

        self.mat_stack.append(self.mat_stack[-1].clone())
        return popmanager()

    def pop_matrix(self):
        """
        Restore the previous transformation
        """
        self.mat_stack.pop()

    def push_style(self):
        """
        Save the current drawing state
        """

        @contextmanager
        def popmanager():
            pass
            try:
                yield
            finally:
                self.pop_style()

        self.draw_states.append(copy.copy(self.draw_states[-1]))
        return popmanager()

    def pop_style(self):
        """
        Restore the previously pushed drawing state
        """
        old = self.draw_states.pop()
        self.draw_states[-1].set(old)

    def push(self):
        """
        Save the current drawing state and transformations
        """

        @contextmanager
        def popmanager():
            pass
            try:
                yield
            finally:
                self.pop()

        self.push_matrix()
        self.push_style()
        return popmanager()

    def pop(self):
        """
        Restore the previously pushed drawing state and transformations
        """
        self.pop_matrix()
        self.pop_style()

    @property
    def _transform(self):
        return self.mat_stack[-1]

    @_transform.setter
    def _transform(self, mat):
        self.mat_stack[-1] = mat

    def _mat(self, M):
        return make_mat(M, self.device, self.dtype)

    def _vec(self, *args):
        return make_vec(*args, device=self.device, dtype=self.dtype)

    def to(self, v):
        """Converts a value or tensor to the appropriate dtype and device"""
        return torch.as_tensor(
            v, dtype=self.dtype, device=self.device
        )  # .to(self.dtype).to(self.device)

    _to = to

    def translate(self, *args):
        """Translate by specifying `x` and `y` offset.

        Arguments:

        - The offset can be specified as an array/list/tensor (e.g `translate([x,y])`
          or as single arguments (e.g. `translate(x, y)`)
        """
        if len(args) == 1:
            p = self._to(args[0])
            x, y = p
        else:
            x, y = [self._to(v) for v in args]
        M = self._mat([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])
        self._transform = self._transform @ M

    def rotate(self, angle):
        """Rotate by `theta` radians (or degrees, depeending on the angle mode)"""
        # angle in rad, differentiable
        angle = self._to(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        M = self._mat([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self._transform = self._transform @ M

    def scale(self, *args):
        """Apply a scaling transformation.

        Arguments:

        - Providing a single number will apply a uniform transformation.
        - Providing a pair of number will scale in the x and y directions.
        - The scale can be specified as an array/list/tensor (e.g `scale([x,y])`
        or as single arguments (e.g. `scale(x, y)`)'''
        """
        if len(args) == 1:
            s = args[0]
            s = self._to(s)
            if is_number(s):
                sx, sy = s, s
            else:
                sx, sy = s
        else:
            sx, sy = [self._to(v) for v in args]

        M = self._mat([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        self._transform = self._transform @ M

    def identity(self):
        self._transform = torch.tensor(np.eye(3), device=self.device, dtype=self.dtype)

    reset_matrix = identity
    # Automatic see above
    # @property
    # def cur_fill(self):
    #     return self.draw_states[-1].cur_fill

    # @cur_fill.setter
    # def cur_fill(self, value):
    #     self.draw_states[-1].cur_fill = value

    # @property
    # def cur_stroke(self):
    #     return self.draw_states[-1].cur_stroke

    # @cur_stroke.setter
    # def cur_stroke(self, value):
    #     self.draw_states[-1].cur_stroke = value

    def _get_stroke_or_fill_color(self):
        if self.cur_stroke is not None:
            return torch.as_tensor(self.cur_stroke)  # * self.color_scale
        if self.cur_fill is not None:
            return torch.as_tensor(self.cur_fill)  # * self.color_scale
        return None

    @property
    def center(self):
        """The center of the canvas (as a 2d numpy array)"""
        return self._to([self._width / 2, self._height / 2])

    @property
    def width(self) -> int:
        """The width of canvas"""
        return self._width

    @property
    def height(self) -> int:
        """The height of canvas"""
        return self._height

    def no_fill(self):
        """Do not fill subsequent shapes"""
        self.fill(None)
        return self

    def no_stroke(self):
        """Do not stroke subsequent shapes"""
        self.stroke(None)
        return self

    def fill_rule(self, rule):
        """Sets the fill rule for complex shapes.

        Arguments:
        - One of `"evenodd"`, `"nonzero"`, or `"winding"`
        """
        self._fill_rule = rule
        return self

    def angle_mode(self, mode):
        mode = mode.lower()
        if not mode in ["degrees", "radians"]:
            raise ValueError("invalid angle mode, use either RADIANS or DEGREES")
        self._angle_mode = mode
        return self

    def _to_radians(self, ang):
        if self._angle_mode == "radians":
            return ang
        return np.radians(ang)

    def _to_degrees(self, ang):
        if self._angle_mode == "degrees":
            return ang
        return np.degrees(ang)

    def _get_color(self, *args):
        if len(args) == 1:
            if not is_number(args[0]):
                x = args[0]
                if len(x) == 4:
                    return self._to(x)
                elif len(x) == 3:
                    return torch.cat([self._to(x), self._vec(1.0)])
                elif len(x) == 2:
                    return self._vec(x[0], x[0], x[0], x[1])
                else:
                    return self._vec(x[0], x[0], x[0], 1.0)
            else:
                return self._vec(args[0], args[0], args[0], 1.0)
        elif len(args) == 2:
            return self._vec(args[0], args[0], args[0], args[1])
        elif len(args) == 3:
            return self._vec(args[0], args[1], args[2], 1.0)
        elif len(args) == 4:
            return self._vec(*args)
        raise ValueError("Invalid arg combination")

    def background(self, *args, reset=True):
        """Clear the canvas with a given color
        Accepts either a tensor with the color components, or single color components (as in `fill`)

        Note that by default this resets the scene (`reset=True`) so any primitive preceding background will be effectively lost during rendering.
        """

        if not len(args):
            raise ValueError("background requires at least one argument")

        # Background clears so we may as well begin
        if reset:
            self.reset()

        if args[0] is None:
            self._bg = None
            return

        clr = self._get_color(*args)[:3]
        self._bg = torch.zeros(
            self.height, self.width, 3, dtype=torch.float32, device=self.device
        )
        self._bg[...] = torch.as_tensor(clr).to(self.dtype).to(self.device)
        return self

    def fill(self, *args):
        """Set the color of the current fill

        Arguments:

        - A single argument specifies a grayscale value, e.g `fill(0.5)` will fill with 50% gray.
        - Two arguments specify grayscale with opacity, e.g. `fill(1.0, 0.5)` will fill with transparent white.
        - TODO Three arguments specify a color depending on the color mode (rgb or hsv)
        - Four arguments specify a color with opacity
        """
        if args[0] is None:
            self.cur_fill = None
        else:
            self.cur_fill = self._get_color(*args)
        return self

    def stroke(self, *args):
        """Set the color of the current stroke

        Arguments:
        - A single argument specifies a grayscale value, e.g. `stroke(255)` will set the stroke to white.
        - Two arguments specify grayscale with opacity, e.g. `stroke(0, 128)` will set the stroke to black with 50% opacity.
        - TODO Three arguments specify a color depending on the color mode (rgb or hsv), e.g. `stroke(255, 0, 0)` will set the stroke to red, when the color mode is RGB
        - Four arguments specify a color with opacity
        """

        if args[0] is None:
            self.cur_stroke = None
        else:
            self.cur_stroke = self._get_color(*args)
        return self

    def stroke_weight(self, w):
        """Set the line width

        Arguments:
        - The width in pixel of the stroke
        """
        self._line_width = w
        return self

    def curve_tightness(self, val):
        """Sets the 'tension' parameter for the curve used when using `curve_vertex`"""
        self._tension = val
        if self.cur_shape is not None:
            self.cur_shape.tension = val
        return self

    def rect_mode(self, mode):
        """Set the "mode" for drawing rectangles.

        Arguments:
        - `mode` (string): can be one of 'corner', 'corners', 'center', 'radius'

        """
        mode = mode.lower()
        if mode not in ["corner", "center", "radius", "corners"]:
            print("rect_mode: invalid mode")
            print("choose one among: corner, center, radius")
            return
        self._rect_mode = mode
        return self

    def ellipse_mode(self, mode):
        """Set the "mode" for drawing rectangles.

        Arguments:
        - `mode` (string): can be one of 'corner', 'center'
        """
        mode = mode.lower()
        if mode not in ["corner", "center", "radius", "corners"]:
            print("rect_mode: invalid mode")
            print("choose one among: corner, center")
            return
        self._ellipse_mode = mode
        return self

    def begin_shape(self):
        """Start building a complex shape. Drawing is deferred until end_shape()."""
        self.cur_shape = Shape(tension=self._tension)
        self.cur_shape.begin_shape()
        self.building_shape = True
        return self

    def end_shape(self, close=False):
        """Finish the shape and draw it."""
        if self.cur_shape is None:
            return
        self.building_shape = False
        self.cur_shape.end_shape(close)
        self._build_shape(self.cur_shape)
        self.cur_shape = None
        return self

    def begin_contour(self):
        """Start a new contour within the currently built shape.
        If no shape is active, a new one is created automatically."""
        if self.cur_shape is None:
            self.cur_shape = Shape(tension=self._tension)
            self.cur_shape.begin_shape()
        self.cur_shape.begin_contour()
        return self

    def end_contour(self, close=False):
        """End the current contour. If not inside a begin_shape/end_shape block,
        the contour is drawn immediately.

        Arguments:

        - `close` (bool, optional): if `True` close the contour

        """
        if self.cur_shape is None:
            return
        self.cur_shape.end_contour(close)
        if not self.building_shape:
            # Called directly, so finalise and draw now
            self.cur_shape.end_shape()
            self._build_shape(self.cur_shape)
            self.cur_shape = None
        return self

    def vertex(self, *args):
        """Add a vertex to current contour

        Input arguments can be in the following formats:

        - `[x, y]`
        - `x, y`
        """
        if self.cur_shape is None:
            raise RuntimeError("vertex() called without begin_shape()")
        self.cur_shape.vertex(*args)
        return self

    def curve_vertex(self, *args):
        """Add a curved vertex to current contour

        Input arguments can be in the following formats:

        - `[x, y]`
        - `x, y`
        """
        if self.cur_shape is None:
            raise RuntimeError("curve_vertex() called without begin_shape()")
        self.cur_shape.curve_vertex(*args)
        return self

    def bezier_vertex(self, *args):
        """Draw a cubic Bezier segment from the current point
        requires a first control point to be already defined with `vertex`.


        Requires three points. Input arguments can be in the following formats:

        - `[x1, y1], [x2, y2], [x3, y3]`
        - `x1, y1, x2, y2, x3, y3`
        """
        if self.cur_shape is None:
            raise RuntimeError("bezier_vertex() called without begin_shape()")
        self.cur_shape.bezier_vertex(*args)
        return self

    def polyline(self, *args, close=False):
        """Draw a polyline (open by default).

        The polyline is specified as either:

        - a list of =[x,y]= pairs (e.g. =[[0, 100], [200, 100], [200, 200]]=)
        - a tensor array with shape =(n, 2)=, representing =n= points (a point for each row and a coordinate for each column)
        - two 1d sequences/tensors, one for each coordinate

        To close the polyline set the named =close= argument to =True=, e.g. =c.polyline(points, close=True)=.
        """
        if len(args) == 1:
            points = self._to(args[0])
        elif len(args) > 1:
            points = torch.vstack([self._to(v) for v in args]).T
        else:
            raise ValueError("Wrong number of arguments")
        self.begin_contour()
        self.cur_shape._polyline(points, close)
        self.end_contour(close)
        return self

    def polygon(self, points):
        """Draw a closed polygon from a sequence of (x,y) points."""
        self.polyline(points, close=True)
        return self

    def multibezier(self, *args, close=False):
        """
        Draw a sequence of connected cubic Bézier curves.


        Input can be

        - a list of =[x,y]= pairs (e.g. =[[0, 100], [200, 100], [200, 200]]=)
        - a tensor array with shape =(n, 2)=, representing =n= control points (a point for each row and a coordinate for each column)
        - two 1d sequences/tensors, one for each coordinate

        To close the curve set the named =close= argument to =True=, e.g. =c.multibezier(points, close=True)=.
        """
        if len(args) == 1:
            points = self._to(args[0])
        elif len(args) > 1:
            points = torch.vstack([self._to(v) for v in args]).T
        else:
            raise ValueError("Wrong number of arguments")
        self.begin_contour()
        self.cur_shape._multibezier(points, close)
        self.end_contour(close)
        return self

    def curve(self, *args, close=False):
        """
        Draw a curve (open by default) using Cardinal spline interpolation.

        Control the tension of the curve using `curve_tightness(...)` with a value between 0 and 1 (default 0.5)
        Input can be

        - a list of =[x,y]= pairs (e.g. =[[0, 100], [200, 100], [200, 200]]=)
        - a tensor array with shape =(n, 2)=, representing =n= control points (a point for each row and a coordinate for each column)
        - two 1d sequences/tensors, one for each coordinate

        To close the curve set the named =close= argument to =True=, e.g. =c.curve(points, close=True)=.
        """
        if len(args) == 1:
            points = self._to(args[0])
        elif len(args) > 1:
            points = torch.vstack([self._to(v) for v in args]).T
        else:
            raise ValueError("Wrong number of arguments")
        self.begin_contour()
        self.cur_shape._curve(points, close)
        self.end_contour(close)
        return self

    def shape(self, obj, close=False):
        """Draw a pre‑built Shape object or a list of polylines (list of lists/arrays).
        For lists, each polyline becomes one contour (open or closed)."""

        if isinstance(obj, Shape):
            if obj in self.shape_to_inds:
                # Create an instance if we are reusing the shape obj
                inds = self.shape_to_inds[obj]
                self._instance_primitives(inds)
            else:
                self._build_shape(obj)
            return

        # Convert polyline lists into a temporary Shape
        if not is_compound(obj):
            obj = [obj]
        tmp_shape = Shape()
        tmp_shape.begin_shape()
        for poly in obj:
            pts = torch.as_tensor(poly)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError("Each polyline must be an Nx2 array-like")
            tmp_shape.polyline(pts, close)
        tmp_shape.end_shape()
        self._build_shape(tmp_shape)
        return self

    def _as_point(self, p):
        """Convert a 2D point to a differentiable tensor while preserving grads."""
        if isinstance(p, (list, tuple)) and len(p) == 2:
            return self._vec(p[0], p[1])
        return self._to(p)

    def _as_size(self, s):
        """Convert a size (scalar or [w,h]) to a differentiable 2D tensor."""
        if isinstance(s, (list, tuple)) and len(s) == 2:
            return self._vec(s[0], s[1])
        t = self._to(s)
        if t.numel() == 1:
            return self._vec(t, t)
        return t

    def rectangle(self, *args, mode=None, radius=None):
        """Draw a rectangle. Alias: =rect=.

        Arguments:
        - =[[x1,y1],[x2,y2]]=            (corners implied)
        - =[x,y], [w,h]=
        - =[x,y], w, h=
        - =x, y, w, h=
        - any of the above with =radius=r= for rounded corners
        """
        # TODO use rectangle primitive for non rounded case
        if mode is None:
            mode = self._rect_mode
        mode = mode.lower()

        # Parse position/size
        if len(args) == 1:
            # corners explicitly given
            p, b = self._to(args[0])
            size = b - p
            mode = "corner"
        elif len(args) == 2:  # [x,y], [w,h]
            p = self._to(args[0])
            size = self._to(args[1])
        elif len(args) == 3:  # [x,y], w, h
            p = self._to(args[0])
            size = self._vec(args[1], args[2])
        elif len(args) == 4:  # x, y, w, h
            p = self._vec(args[0], args[1])
            size = self._vec(args[2], args[3])
        else:
            raise ValueError("rectangle: wrong number of arguments")

        # Convert modes to top-left corner + full width/height
        if mode == "corners":
            size = size - p  # Actually size is max corner
        elif mode == "center":
            p = p - size / 2
        elif mode == "radius":
            p = p - size
            size = size * 2

        x, y = p
        w, h = size

        if radius is None:
            prim = pydiffvg.Rect(
                p,
                p + size,
                stroke_width=self._to(self._line_width),
            )
            self._add_primitives([prim])
            # pts = self._mat(
            #     [
            #         [x, y],
            #         [x + w, y],
            #         [x + w, y + h],
            #         [x, y + h],
            #     ]
            # )

            # self.polyline(pts, close=True)
        else:
            r = torch.min(self._to(radius), torch.min(w, h) / 2)
            k = self._to(0.5522847498)
            o = r * (1 - k)

            pts = self._mat(
                [
                    # top-right corner
                    [x + r, y],
                    [x + w - o, y],
                    [x + w, y + o],
                    [x + w, y + r],
                    # bottom-right corner
                    [x + w, y + h - o],
                    [x + w - o, y + h],
                    [x + w - r, y + h],
                    # bottom-left corner
                    [x + o, y + h],
                    [x, y + h - o],
                    [x, y + h - r],
                    # top-left corner
                    [x, y + o],
                    [x + o, y],
                ]
            )
            self.multibezier(pts, close=True)
        return self

    rect = rectangle

    def square(self, *args, mode=None):
        """Draw a square."""
        if mode is None:
            mode = self._rect_mode
        if mode == "corners":
            mode = "corner"

        if len(args) == 2:
            self.rectangle(args[0], self._vec(args[1], args[1]), mode=mode)
        elif len(args) == 3:
            self.rectangle(args[0], args[1], args[2], args[2], mode=mode)
        else:
            raise ValueError("square: wrong number of arguments")
        return self

    def quad(self, *args):
        """Draw a quadrilateral."""
        if len(args) == 4:
            points = args
        elif len(args) == 8:
            points = [[args[i * 2], args[i * 2 + 1]] for i in range(4)]
        else:
            raise ValueError("quad: wrong number of arguments")
        self.polygon(points)
        return self

    def triangle(self, *args):
        """Draw a triangle."""
        if len(args) == 3:
            points = args
        elif len(args) == 6:
            points = [[args[i * 2], args[i * 2 + 1]] for i in range(3)]
        else:
            raise ValueError("triangle: wrong number of arguments")
        self.polygon(points)
        return self

    def ellipse(self, *args, mode=None):
        """Draw an ellipse.

        Arguments:
        - =[cx,cy], [w,h]=
        - =[cx,cy], w=              (circle)
        - =[cx,cy], w, h=
        - =cx, cy, w=               (circle)
        - =cx, cy, w, h=
        """
        # Contructs the ellipse with Beziers since DiffVG does not support
        # outlines for ellipse primitives. One more efficient option would be
        # to check the stroke and create an ellipse primitive if stroke is not set
        if mode is None:
            mode = self._ellipse_mode
        mode = mode.lower()

        if len(args) == 2:
            center = self._to(args[0])
            size = self._as_size(args[1])
        elif len(args) == 3:
            if is_number(args[0]):
                center = self._vec(args[0], args[1])
                size = self._vec(args[2], args[2])

            else:
                center = self._to(args[0])
                size = self._vec(args[1], args[2])
        elif len(args) == 4:
            center = self._vec(args[0], args[1])
            size = self._vec(args[2], args[3])
        else:
            raise ValueError("ellipse: wrong number of arguments")

        if mode == "corners":
            x1, y1 = center
            x2, y2 = size
            center, size = (center + size) / 2, torch.abs(size - center)
        elif mode == "corner":
            center = center + size / 2
        elif mode == "radius":
            size *= 2

        cx, cy = center
        rx, ry = size[0] / 2, size[1] / 2

        # 4-segment cubic Bézier approximation of an ellipse
        k = self._to(0.5522847498)
        pts = self._mat(
            [
                [cx + rx, cy],
                [cx + rx, cy + ry * k],
                [cx + rx * k, cy + ry],
                [cx, cy + ry],
                [cx - rx * k, cy + ry],
                [cx - rx, cy + ry * k],
                [cx - rx, cy],
                [cx - rx, cy - ry * k],
                [cx - rx * k, cy - ry],
                [cx, cy - ry],
                [cx + rx * k, cy - ry],
                [cx + rx, cy - ry * k],
            ]
        )

        self.multibezier(pts, close=True)
        return self

    def circle(self, *args, mode=None):
        """Draw a circle.

        Arguments:
        - =[cx,cy], r=
        - =cx, cy, r=
        """
        if mode is None:
            mode = self._ellipse_mode
        mode = mode.lower()

        if len(args) == 2:
            center = self._to(args[0])
            size = self._to(args[1])
        elif len(args) == 3:
            center = self._vec(args[0], args[1])
            size = self._to(args[2])
        else:
            raise ValueError("circle: wrong number of arguments")

        if mode == "radius":
            size = size * 2

        if mode == "corner":
            center = center + size

        prim = pydiffvg.Circle(
            radius=size / 2, center=center, stroke_width=self._to(self._line_width)
        )
        self._add_primitives([prim])

        # self.ellipse(center, size, size, mode="center")
        return self

    ###############################################
    # Scene management

    def _add_primitives(self, primitives):
        """Add new primitives for rendering"""
        ind = len(self.primitives)
        self.primitives += primitives
        shape_ids = list(range(ind, ind + len(primitives)))
        self._instance_primitives(shape_ids)
        return shape_ids

    def _instance_primitives(self, shape_ids):
        """Create groups for given primitive indices"""
        fill_color = None
        if self.cur_fill is not None:
            fill_color = self.cur_fill.to(self.device)
        stroke_color = None
        if self.cur_stroke is not None:
            stroke_color = self.cur_stroke.to(self.device)

        group = pydiffvg.ShapeGroup(
            shape_ids=torch.as_tensor(shape_ids).to(torch.int64).to(self.device),
            use_even_odd_rule=self._fill_rule == "evenodd",  # evenodd',
            fill_color=fill_color,
            stroke_color=stroke_color,
        )
        group.shape_to_canvas = self._transform.to(self.device)
        self.groups.append(group)

    def _build_shape(self, shape):
        primitives = shape.build(self)
        inds = self._add_primitives(primitives)
        # store for instancing if shape is called with same object multiple times
        self.shape_to_inds[shape] = inds

    def line(self, *args):
        """Draws a line between two points

        Input arguments can be in the following formats:

         - `a, b` (Two points specified as lists/tuples/arrays/tensors
         - `x1, y1, x2, y2`, a sequence of numbers, one for each coordinate
        """
        if len(args) == 2:
            a, b = [self._to(v) for v in args]
        elif len(args) == 4:
            ax, ay, bx, by = [self._to(v) for v in args]
            a = torch.stack([ax, ay])
            b = torch.stack([bx, by])
        else:
            raise ValueError("line: Unexpected number of arguments")

        self.polyline(torch.vstack([a, b]))

    def render(
        self, prefiltering=False, num_samples=2, seed=0, sdf=False, auto_reset=True
    ):
        """Render the canvas output

        Arguments:
        - `prefiltering`: if `True`, uses an anti‑aliasing prefilter. Produces crisper lines, but does not support variable width strokes and produces artefacts in some cases.
        - `num_samples`: number of x and y samples for Montecarlo boundary sampling in DiffVG.
        - `sdf`: if `True`, outputs a signed distance field.
        - `autoreset`: if `True` (default) the DiffVG scene is reset after rendering, so that the next draw calls rebuild it
        """
        if prefiltering:
            num_samples = 1

        if self._bg is not None:
            bg = torch.as_tensor(self._bg).to(self.dtype).to(self.device)
            if len(bg.shape) == 2:
                bg = bg[:, :, np.newaxis]
                bg = bg.repeat(1, 1, 3)
            h, w, _ = bg.shape
        else:
            w, h = self.width, self.height

        if not self.primitives:
            self.img = bg
            return self.img

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w,
            h,
            self.primitives,
            self.groups,
            use_prefiltering=prefiltering,
            output_type=pydiffvg.OutputType.sdf if sdf else pydiffvg.OutputType.color,
        )
        try:
            img = pydiffvg.RenderFunction.apply(
                w, h, num_samples, num_samples, seed, None, *scene_args
            )
        except RuntimeError as e:
            print("RUNTIME ERROR IN RENDER")
            print("Possibly wrong dtype in geometry, needs to be float32")
            raise (e)

        if self._bg is not None:
            img = img[:, :, 3:4] * img[:, :, :3] + bg * (1 - img[:, :, 3:4])
            # Convert to RGB only with background otherwise keep alpha
            img = img[:, :, :3]

        self.img = img

        if auto_reset:
            self.reset()
        return img

    def get_image(self):
        """Return the rendered output as a PIL Image"""
        assert self.img is not None
        img = self.img.detach().cpu().numpy()
        return Image.fromarray((img * 255).astype(np.uint8))

    def get_array(self):
        """Return the rendererd output as a NumPy array"""
        assert self.img is not None
        img = self.img.detach().cpu().numpy()
        return img

    def _repr_png_(self):
        """Tells Jupyter to render this object as a PNG image."""
        import io

        byte_arr = io.BytesIO()
        self.get_image().save(byte_arr, format="PNG")
        return byte_arr.getvalue()

    ## Variable management
    def var(self, v, group_name="", grad=True, id=None):
        """
        Return a tensor for the given variable, input can be a tensor or a sequence

        If `group_name` is provided, the variable is cached, allowing it to be reused across
        multiple drawing operations without recreating the tensor, e.g
        ```python
        c.curve(c.var([[0,0], [100,0], [100,100]], 'pts'))
        c.curve(c.var([[20,0], [10,40], [100,100]], 'pts'))
        ```
        This will cache two tensors that can be retrieved as a list with `c.get_vars('pts')`.

        Modifying these tensors will modify the values used in subsequent calls to the
        same drawing sequence, meaning we can optimize the variable in a loop.

        > NOTE: While this is handy to avoid setup code, keep in mind that alterning the rendering order after these variables have been cached
        will result in unexpected behaviors.
        """
        v = torch.as_tensor(v).to(self.dtype).to(self.device)
        v.requires_grad = grad
        if group_name:
            if id is None:
                # Automatically get id for each draw sequence
                # Simply the count for a given group
                id = self._var_counters[group_name]
                self._var_counters[group_name] += 1

            var_id = self._var_id(group_name, id)
            if var_id in self._id_to_var:
                return self._id_to_var[var_id]

            self._vars[group_name].append(v)
            self._id_to_var[var_id] = v

        return v

    def get_vars(self, group_name):
        """Get vars for a given group name (alias `vars`)"""
        if group_name not in self._vars:
            return []
        return self._vars[group_name]

    def get_points(self):
        """Get points from DiffVG primitives"""
        return [prim.points for prim in self.primitives]

    vars = get_vars

    def clear_vars(self):
        self._vars = defaultdict(list)
        self._var_counters = defaultdict(int)  # Use for auto id
        self._id_to_var = {}

    def _var_id(self, name, id):
        return f"{name}_{id}"

    def to_canvas(self, save_background=True):
        from py5canvas import Canvas

        c = Canvas(self.width, self.height, save_background=save_background)
        c.color_mode("rgb", 1.0)

        if thick:
            c.no_stroke()
            c.fill(0)
        else:
            c.stroke(0)
            c.no_fill()

        for g in self.groups:
            with c.push_matrix():
                c.apply_matrix(npy(g.shape_to_canvas))
                for i in npy(g.shape_ids):
                    prim = self.primitives[i]
                    if isinstance(prim, pydiffvg.Path):
                        Cp = npy(prim.points)
                        if prim.degree == 3:
                            c.multibezier(Cp, close=prim.is_closed)
                        elif prim.degree == 1:
                            c.polyline(Cp)
                        else:
                            raise ValueError(
                                "Degrees other than 1 and 3 not supported yet"
                            )

        return c


class Shape:
    """
    Holds a list of contours, each contour being a sequence of drawing commands.
    Mirrors Processing's PShape: use begin_shape()/end_shape() to construct.
    """

    def __init__(self, tension=0.5):
        self.tension = tension
        self.contours = []  # list of contour command lists
        self._built = False
        self._shape_active = False  # True between begin_shape()/end_shape()
        self.reset()

    def reset(self):
        self._contour = []  # list of commands for the contour being built
        self._num_ctrl = []
        self._curve_points = []  # pending Catmull‑Rom points for curve_vertex
        self._spline_start = None  # first point of the current spline (move-to)

    def _ensure_mutable(self):
        if self._built:
            raise ValueError(
                "Cannot add contours to a shape that has been built, clone it instead"
            )

    def begin_shape(self):
        """Start building the shape. Clears any previous geometry."""
        self._ensure_mutable()
        self.reset()
        self._shape_active = True
        self.contours = []

    def end_shape(self, close=False):
        """
        Finish building the shape.
        If close is True, the last contour is closed before finalising.
        """
        self._ensure_mutable()
        if self._shape_active:
            if self._contour or self._curve_points:
                self.end_contour(close)
            self._shape_active = False

    def begin_contour(self):
        """Start a new contour. Must be called after begin_shape()."""
        self._ensure_mutable()
        if not self._shape_active:
            raise RuntimeError("begin_shape() must be called before begin_contour()")
        self.reset()

    def end_contour(self, close=False):
        """Finish the current contour. If close=True, the contour is closed."""
        self._ensure_mutable()
        self._flush_spline(close=close)
        if not self._contour:
            return
        pts = torch.vstack(self._contour)
        n_ctrl = torch.tensor(self._num_ctrl, dtype=torch.int32)
        self.contours.append((pts, n_ctrl, close))
        self.reset()

    def vertex(self, *args):
        """Add a straight vertex ."""
        self._ensure_mutable()
        if len(args) > 1:
            x = torch.stack([torch.as_tensor(v) for v in args])
        else:
            x = torch.as_tensor(args[0])
        self._start_contour_if_needed()
        self._flush_spline()
        if self._contour:
            self._num_ctrl += [0]
        self._contour.append(x)

    def curve_vertex(self, *args):
        """Add a curved vertex (Catmull Rom spline)."""
        self._ensure_mutable()
        if len(args) > 1:
            x = torch.stack([torch.as_tensor(v) for v in args])
        else:
            x = torch.as_tensor(args[0])

        self._start_contour_if_needed()
        if not self._curve_points:
            if self._contour and (not self._num_ctrl or self._num_ctrl[-1] == 0):
                self._spline_start = self._contour[-1][-1].clone()
            else:
                self._spline_start = None
        self._curve_points.append(x)

    def bezier_vertex(self, *args):
        """Add a cubic Bézier vertex; three control points."""
        self._ensure_mutable()
        if len(args) == 1:
            pts = torch.as_tensor(args[0])
        elif len(args) == 3:
            pts = torch.vstack([torch.as_tensor(v) for v in args])
        elif len(args) == 6:
            pts = torch.stack([torch.as_tensor(v) for v in args]).reshape(3, 2)
        else:
            raise ValueError("bezier_vertex expects 3 points or 6 scalars")

        self._start_contour_if_needed()
        self._flush_spline()
        self._contour.append(pts)
        self._num_ctrl += [2]

    def _polyline(self, points, closed):
        self._ensure_mutable()

        self._contour.append(torch.as_tensor(points))
        nseg = len(points) - 1
        if closed:
            nseg += 1
        self._num_ctrl += [0] * nseg

    def polyline(self, points, close=False):
        """Add a contour of straight line segments from a sequence of (x,y) points."""
        if not self._shape_active:
            self.begin_shape()  # temporary activation for standalone use
        self.begin_contour()
        self._polyline(points, close)
        self.end_contour(close)

    def _multibezier(self, points, close):
        self._ensure_mutable()
        num_segs = num_bezier(points, close)
        self._contour.append(torch.as_tensor(points))
        self._num_ctrl += [2] * num_segs

    def multibezier(self, points, close=False):
        """
        Add a contour of cubic Bézier segments.
        TODO handle concatenation
        """
        if not self._shape_active:
            self.begin_shape()
        self.begin_contour()
        self._multibezier(points, close)
        self.end_contour(close)

    def _curve(self, points, close):
        self._multibezier(
            cardinal_spline(torch.as_tensor(points), self.tension, close), close
        )

    def curve(self, points, close=False):
        """
        Add a contour of smooth Cardinal spline segments.
        points: sequence of (x,y) knots.
        TODO handle concatenation
        """
        self.begin_contour()
        self._curve(points, close)
        self.end_contour(close)

    def _start_contour_if_needed(self):
        if not self._contour:
            self.begin_contour()

    def _flush_spline(self, close=False):
        if not self._curve_points or len(self._curve_points) < 2:
            self._curve_points = []
            self._spline_start = None
            return

        # Build full points list: include previous anchor if available
        pts = torch.vstack(self._curve_points)
        if self._spline_start is not None:
            pts = torch.vstack([self._spline_start] + pts)

        cp = cardinal_spline(pts, self.tension, closed=close)

        if self._contour:
            cp = cp[1:]
            m = len(cp) // 3
        else:
            m = (len(cp) - 1) // 3  # TODO test me

        if m <= 0:
            print("Invalid number of control points for spline")
            self._curve_points = []
            self._spline_start = None
            return

        self._contour.append(cp)
        self._num_ctrl += [2] * m
        self._curve_points = []
        self._spline_start = None

    def clone(self):
        new = Shape(tension=self.tension)
        new.contours = [ctr for ctr in self.contours]
        new.reset()
        return new

    def build(self, c):
        """Build diffvg primitives"""
        shapes = []
        for ctr in self.contours:
            if isinstance(ctr, tuple):
                pts, nctrl, closed = ctr
                pts = pts.to(dtype=c.dtype, device=c.device)
                if pts.shape[1] > 2:
                    w = pts[:, 2].contiguous()
                    pts = pts[:, :2].contiguous()
                else:
                    w = torch.as_tensor(c._line_width, device=c.device)

                path = pydiffvg.Path(
                    num_control_points=nctrl.to(device=c.device),
                    points=pts,
                    stroke_width=w,
                    is_closed=closed,
                    use_distance_approx=False,
                )
                shapes.append(path)
            else:  # Assume a diffVg object added externally by canvas
                shapes.append(ctr)
        self._built = True
        return shapes


def cardinal_spline(Q, c, closed=False):
    """Cardinal spline interpolation for a sequence of values"""
    isnp = isinstance(Q, np.ndarray)

    if closed:
        if isnp:
            Q = np.vstack([Q, Q[0:1]])
        else:
            Q = torch.concat([Q, Q[0:1]])
    n = len(Q)
    D = []
    for k in range(1, n - 1):
        # Assuming uniform parametrisation here
        d = (1 - c) * (Q[k + 1] - Q[k - 1])
        D.append(d)
    if closed:
        d1 = dn = (1 - c) * (Q[1] - Q[-2])
    else:
        d1 = (1 - c) * (Q[1] - Q[0])
        dn = (1 - c) * (Q[-1] - Q[-2])
    D = [d1] + D + [dn]
    P = [Q[0]]
    for k in range(1, n):
        p1 = Q[k - 1] + D[k - 1] / 3
        p2 = Q[k] - D[k] / 3
        p3 = Q[k]
        P += [p1, p2, p3]

    if closed:
        P = P[:-1]
    if isnp:
        return np.vstack(P)
    return torch.vstack(P)


# Bot vectorization
def cardinal_spline(Q, c, closed=False):
    """Vectorized cardinal spline interpolation for a torch tensor of points.

    Q: (n, d) tensor of n points (d>=1).
    Returns the cubic Bezier control points (3n-2 for open, 3n for closed).
    """
    n = Q.shape[0]
    t = 1.0 - c

    if closed:
        # D[k] = t * (Q[k+1] - Q[k-1]) with periodic wraparound
        D = t * (torch.roll(Q, -1, 0) - torch.roll(Q, 1, 0))
        starts = Q
        ends = torch.roll(Q, -1, 0)
        D_starts = D
        D_ends = torch.roll(D, -1, 0)
        n_seg = n
    else:
        D = torch.empty_like(Q)
        if n > 2:
            D[1:-1] = t * (Q[2:] - Q[:-2])
        D[0] = t * (Q[1] - Q[0])
        D[-1] = t * (Q[-1] - Q[-2])
        starts = Q[:-1]
        ends = Q[1:]
        D_starts = D[:-1]
        D_ends = D[1:]
        n_seg = n - 1

    # Two intermediate Bezier handles per segment, plus the segment endpoint.
    c1 = starts + D_starts / 3.0
    c2 = ends - D_ends / 3.0

    # Interleave: [Q[0], c1[0], c2[0], ends[0], c1[1], c2[1], ends[1], ...]
    P = torch.empty((1 + 3 * n_seg, *Q.shape[1:]), dtype=Q.dtype, device=Q.device)
    P[0] = Q[0]
    P[1:] = torch.stack([c1, c2, ends], dim=1).reshape(-1, *Q.shape[1:])

    if closed:
        P = P[:-1]  # drop the duplicated closing point
    return P


def is_compound(S):
    """Returns True if S is a compound polyline,
    a polyline is represented as a list of points, or a ndarray/tensor with as many rows as points"""
    if type(S) != list:
        return False
    if type(S) == list:
        if not S:
            return True
        for P in S:
            try:
                if is_number(P[0]):
                    return False
            except IndexError:
                pass
        return True
    if (isinstance(S[0], torch.Tensor) or isinstance(S[0], np.ndarray)) and len(
        S[0].shape
    ) > 1:
        return True
    return False


def split_cubic(bez, t):
    p1, p2, p3, p4 = bez
    p12 = (p2 - p1) * t + p1
    p23 = (p3 - p2) * t + p2
    p34 = (p4 - p3) * t + p3
    p123 = (p23 - p12) * t + p12
    p234 = (p34 - p23) * t + p23
    p1234 = (p234 - p123) * t + p123
    return np.vstack([p1, p12, p123, p1234]), np.vstack([p1234, p234, p34, p4])


def beziers_to_chain(beziers):
    n = len(beziers)
    chain = []
    for i in range(n):
        chain.append(list(beziers[i][:-1]))
    chain.append([beziers[-1][-1]])
    return torch.vstack(sum(chain, []))


def chain_to_beziers(chain, degree=3):
    """Convert Bezier chain to list of curve segments (4 control points each)"""
    num = num_bezier(chain.shape[0], degree)
    beziers = []
    for i in range(num):
        beziers.append(chain[i * degree : i * degree + degree + 1, :])
    return beziers


def beziers_to_chain(beziers):
    """Convert list of Bezier curve segments to a piecewise bezier chain (shares vertices)"""
    n = len(beziers)
    chain = []
    for i in range(n):
        chain.append(list(beziers[i][:-1]))
    chain.append([beziers[-1][-1]])
    return np.array(sum(chain, []))


def tangent_angle(bez):
    d0 = bez[1, :2] - bez[0, :2]
    d1 = bez[3, :2] - bez[2, :2]
    d0 = d0 / (np.linalg.norm(d0) + 1e-8)
    d1 = d1 / (np.linalg.norm(d1) + 1e-8)
    c = np.clip(np.dot(d0, d1), -1.0, 1.0)
    return np.arccos(c)


def subdivide_adaptive_bezier(
    bez,
    thresh,
    depth,
    max_depth,
):
    if depth >= max_depth:
        return [bez]

    if not tangent_angle(bez) > thresh:
        return [bez]

    left, right = split_cubic(bez, 0.5)

    return subdivide_adaptive_bezier(
        left,
        thresh,
        depth + 1,
        max_depth,
    ) + subdivide_adaptive_bezier(
        right,
        thresh,
        depth + 1,
        max_depth,
    )


def bezier_mat(p, t, deriv=0):
    n = p + 1
    if deriv > 0:
        return np.diff(np.eye(n), 1) @ bezier_mat(p - 1, t, deriv - 1)
    B = np.vstack([bernstein(p, i)(t) for i in range(n)])
    return B


def bernstein(n, i):
    bi = math.comb(n, i)  # binom(n, i)
    return lambda t, bi=bi, n=n, i=i: bi * t**i * (1 - t) ** (n - i)


def bezier(P, t, d=0):
    """Bezier curve of degree len(P)-1. d is the derivative order (0 gives positions)"""
    n = P.shape[0] - 1
    if d > 0:
        Q = np.diff(P, axis=0) * n
        return bezier(Q, t, d - 1)
    B = np.vstack([bernstein(n, i)(t) for i, p in enumerate(P)])
    return (P.T @ B).T


def num_bezier(n_ctrl, degree=3):
    if type(n_ctrl) == np.ndarray:
        n_ctrl = len(n_ctrl)
    return int((n_ctrl - 1) / degree)


def thick_bezier_envelope(Cp, subd_thresh=50):
    """Approximate envelope of a Bezier chain with varying width
    Subdivides the segments based depending on tangent angle
    and offsets along uniformly sampled normals.
    Will under-estimate the offset for high curvature segments but works
    reasonably ok with Beziers derived from splines
    """
    from numpy.linalg import norm

    if Cp.shape[1] < 3:
        raise ValueError("Each control point should have radius!")
    beziers = chain_to_beziers(Cp)

    subd = []
    thresh = np.radians(subd_thresh)
    for bez in beziers:
        subd.extend(subdivide_adaptive_bezier(bez, thresh, 0, 5))
    beziers = subd

    L, R = [], []
    t = np.linspace(0, 1, 4)
    for bez in beziers:
        Pr = bezier(bez, t)  # uniform samples
        P = bez[:, :2]  # Offset control points
        r = Pr[:, 2:]  # Use radius samples for offset
        # tangents and normals
        D = bezier(bez[:, :2], t, d=1)
        N = np.vstack([-D[:, 1], D[:, 0]]).T
        N = N / np.linalg.norm(N, axis=1, keepdims=True)
        offset = r * N
        L.append(P - offset)
        R.append(P + offset)
    return beziers_to_chain(L), beziers_to_chain(R)


def num_bezier(n_ctrl, closed=False, degree=3):
    if not is_number(n_ctrl):
        n_ctrl = len(n_ctrl)
    if closed:
        n_ctrl += 1
    return int((n_ctrl - 1) / degree)


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        # DiffVG does not work well with ARM
        return torch.device("cpu")
