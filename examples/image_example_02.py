#!/usr/bin/env python3
import os
import torch
import matplotlib.pyplot as plt
from py5diff import DiffCanvas
from PIL import Image
from tqdm import tqdm
import numpy as np

target_img = Image.open('./spock256.jpg')
w, h = target_img.size

c = DiffCanvas(w, h)

def draw(c):
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

    c.render(prefiltering=True)
    return c.img

# Initial image and target
draw(c)
initial_img = c.get_image()
target = c.to(np.array(target_img.convert('L'))/255)

# Optimization loop
optimizers = [
    torch.optim.Adam(c.get_vars('offset'), lr=1.0),
]

for step in tqdm(range(250), 'Opt progress:'):
    for opt in optimizers:
        opt.zero_grad()
    img = draw(c)
    loss = (c.img.mean(dim=-1) - target).abs().mean() # L1
    loss.backward()
    for opt in optimizers:
        opt.step()

# Display
plt.figure(figsize=(9,4)) 
plt.subplot(131)
plt.title('init'); plt.axis('off')
plt.imshow(initial_img)

plt.subplot(132)
plt.title('target'); plt.axis('off')
plt.imshow(target_img)

plt.subplot(133)
plt.title('optimizied'); plt.axis('off')
plt.imshow(c.get_image())
plt.tight_layout()
plt.savefig('image_example_01.png')
plt.show()


