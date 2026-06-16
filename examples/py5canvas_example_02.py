#!/usr/bin/env python3
from py5canvas import *
from slimgui import imgui, implot

from importlib import reload
import py5diff
from py5diff import diff_canvas as dc
reload(dc)
reload(py5diff)

import torch
import matplotlib.pyplot as plt

from py5diff import DiffCanvas, CanvasOptimizer
from PIL import Image
import numpy as np

target_img = Image.open('./spock256.jpg')
w, h = target_img.size

class MyCanvasOpt(CanvasOptimizer):
    def draw(self, c):
        c.background(1.0)
        c.stroke(0); c.no_fill()
        c.stroke_weight(2.0)
        n_rows = 25
        n_pts = 30
        h = (c.height / n_rows)*0.1
        for row_y in np.linspace(0, c.height, n_rows+2)[1:-1]:
            x = np.linspace(0, c.width, n_pts)
            y = row_y + c.var(np.random.uniform(-h, h, n_pts), 'offset')
            c.curve(x, y)
                    
        c.render(prefiltering=True, num_samples=2)
        return c.img

    def postprocess(self, c):
        with torch.no_grad():
            for v in c.get_vars('size'):
                v.data.clamp_(2, 100)
            for v in c.get_vars('color'):
                v.data.clamp_(0, 1)
                
    def setup(self, c):
        self.optimizers = [
                            torch.optim.Adam(c.get_vars('offset'), lr=1.0),
        ]
        self.target = self.c.to(np.array(target_img.convert('L'))/255)
        
    def loss(self, img):
        loss = (img.mean(dim=-1) - self.target).abs().mean()
        return loss

opt = MyCanvasOpt(w, h)

def parameters():
    return {'foo':False}

def setup():
    create_canvas(w, h)
    color_mode('rgb', 1.0)
    
def gui():
    if imgui.button('Run'):
        opt.run(500)
    pass

def draw():
    background(0)
    opt.step()
    image(opt.get_image())
    
run()
