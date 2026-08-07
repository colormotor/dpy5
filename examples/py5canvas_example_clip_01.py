#!/usr/bin/env python3
from py5canvas import *
from slimgui import imgui, implot

from importlib import reload
import dpy5
from dpy5 import diff_canvas as dc

reload(dc)
reload(dpy5)

import torch
import matplotlib.pyplot as plt

from dpy5 import DiffCanvas, CanvasOptimizer, losses, utils
from PIL import Image
import numpy as np


w, h = 400, 400  # target_img.size
target_img = Image.open("./bunny.jpg").resize((w, h))


class MyCanvasOpt(CanvasOptimizer):
    def init(self, c):
        pass

    def draw(self, c):
        n_rows = 15  # 20
        n_cols = 15  # 20

        pad = 30
        xx, yy = np.meshgrid(
            np.linspace(pad, c.width - pad, n_cols),
            np.linspace(pad, c.height - pad, n_rows),
        )
        Xs = c.var(xx, "coords")
        Ys = c.var(yy, "coords")

        c.background(1.0).stroke(0, 0.8).no_fill()
        c.stroke_weight(1.0)
        f = c.curve
        # f = c.polyline
        pts = []
        for i in range(1, n_rows - 1):
            pts.append(torch.vstack([Xs[i, :], Ys[i, :]]).T)
            f(pts[-1])

        pts = torch.vstack(pts)
        radii = 2 + torch.sigmoid(c.var(np.zeros(len(pts)), "radii")) * 50
        c.no_stroke()
        for i, p in enumerate(pts):
            c.no_stroke().fill(0, 0.3)
            c.circle(p, radii[i])
            c.fill(0)
            c.circle(p, 5)

        c.render(prefiltering=False, num_samples=2)

        return c.img

    def postprocess(self, c):
        with torch.no_grad():
            pass

    def setup(self, c):
        self.optimizers = [
            torch.optim.Adam(c.get_vars("coords"), lr=0.5),
            torch.optim.Adam(c.get_vars("radii"), lr=0.1),
        ]
        self.losses = utils.MultiLoss()
        self.losses.add(
            "clip",
            losses.CLIPVisualLoss(
                semantic_w=0.0,
                clip_model="TeCoA4",  # "CLIPAG"
                blur_sigma=None,
            ),
            500.0,
            (None, "target_img"),
        )

    def loss(self, img):
        return self.losses(
            img,
            target_img=target_img,
        )


opt = MyCanvasOpt(w, h, 300)


def parameters():
    return {"dummy": False}


def setup():
    create_canvas(w, h)
    color_mode("rgb", 1.0)

    if sketch.grabbing:
        opt.run()


def gui():
    if imgui.button("Run"):
        opt.run()
    pass


def draw():
    background(0)
    opt.step()
    image(opt.get_image())


run()
