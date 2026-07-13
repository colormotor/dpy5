#!/usr/bin/env python3
"""
Examle showing how to use some helpers:
 - the `CanvasOptimizer` wrapper
 - the `utils.MultiLoss` helper to combine different losses
 - the `utils.show_animation` function to visualize optimization progress
"""

from jupyter_core.command import _path_with_self
import os
import torch
import matplotlib.pyplot as plt
from dpy5 import DiffCanvas, CanvasOptimizer, utils, losses

from PIL import Image
from tqdm import tqdm
import numpy as np

class MyCanvasOpt(CanvasOptimizer):
    def draw(self, c):
        c.background(1.0)
        c.stroke(0)
        c.no_fill()
        c.stroke_weight(2.0)
        n_rows = 25
        n_pts = 30
        h = (c.height / n_rows) * 0.1
        for row_y in np.linspace(0, c.height, n_rows + 2)[1:-1]:
            x = np.linspace(0, c.width, n_pts)
            y = row_y + c.var(np.random.uniform(-h, h, n_pts), "offset")
            c.curve(x, y)

        c.render(prefiltering=True, num_samples=2)
        return c.img

    def postprocess(self, c):
        with torch.no_grad():
            for v in c.get_vars("size"):
                v.data.clamp_(2, 100)
            for v in c.get_vars("color"):
                v.data.clamp_(0, 1)

    def setup(self, c):
        self.optimizers = [
            torch.optim.Adam(c.get_vars("offset"), lr=1.0),
        ]
        self.losses = utils.MultiLoss()
        self.losses.add(
            "clip", losses.CLIPVisualLoss(semantic_w=0), 1.0, (None, "target_img")
        )
        # self.losses.add("mse", losses.MultiscaleMSELoss(), 1.0, (None, "target_img"))

    def loss(self, img):
        return self.losses(img, target_img=target_img)


# Target image and canvas optimizer
target_img = Image.open("./spock256.jpg")
w, h = target_img.size

opt = MyCanvasOpt(w, h, 300)
opt.run()

# Create figure and run animation
fig = plt.figure(figsize=(8, 8))


def frame(step):
    plt.title(f"Step {step + 1} of {opt.num_opt_steps}")
    opt.step()
    plt.imshow(opt.get_image())


movie_file = ""
utils.show_animation(fig, frame, opt.num_opt_steps, filename=movie_file)
