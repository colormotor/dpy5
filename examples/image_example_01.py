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
    c.fill(0, 0.2)
    c.stroke(0)
    c.rect_mode('center')
    c.stroke_weight(0.5)
    n = 10
    for y in np.linspace(0, c.height, n+2)[1:-1]:
        for x in np.linspace(0, c.width, n+2)[1:-1]:
            c.push_matrix()
            c.translate(x, y)
            c.rotate(c.var(0.0, 'rot'))
            c.rectangle([0, 0], c.var([c.width/n-5, c.width/n-5], 'size'))
            c.pop_matrix()

    c.render(prefiltering=False, num_samples=2)
    return c.img

# Initial image and target
draw(c)
initial_img = c.get_image()
target = c.to(np.array(target_img.convert('L'))/255)

# Optimization loop
optimizers = [
    torch.optim.Adam(c.get_vars('size'), lr=1.0),
    torch.optim.Adam(c.get_vars('rot'), lr=0.1),
]

for step in tqdm(range(250), 'Opt step:'):
    for opt in optimizers:
        opt.zero_grad()
    
    img = draw(c)
    loss = (c.img.mean(dim=-1) - target).abs().mean() # L1
    loss.backward()
    for opt in optimizers:
        opt.step()

plt.figure(figsize=(12,4)) 
plt.subplot(131)
plt.title('init'); plt.axis('off')
plt.imshow(initial_img)

plt.subplot(132)
plt.title('target'); plt.axis('off')
plt.imshow(target_img)

plt.subplot(133)
plt.title('optimizied'); plt.axis('off')
plt.imshow(c.get_image())
plt.savefig('image_example_01.png')
plt.show()


