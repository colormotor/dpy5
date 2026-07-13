# $D\text{py}5$ - Processing-like Differentiable Vector Graphics

`dpy5` provides a Processing-inspired API (e.g., `push()`, `pop()`, `fill()`, `stroke()`, `line()`, `curve()`) for building *differentiable* 2D vector scenes. Under the hood it uses [pydiffvg](https://github.com/BachiLi/diffvg) and [PyTorch](https://pytorch.org), so all parameters are tensors and gradients can flow through the rendering process. It provides an "immediate mode" API on top of DiffVG, making it easier to experiment and build geometry through the composition of differentiable operations.

`dpy5` can be used standalone in Python scripts or Jupyter notebooks, but its API is almost identical to [Py5canvas](https://github.com/colormotor/py5canvas), so it can also be used alongside it to create sketches that leverage differentiable rasterization.
## Installation

Prerequisites:

- [Install pytorch](https://pytorch.org/get-started/locally/)
- [Clone and install DiffVG locally](https://github.com/BachiLi/diffvg)

Install locally by cloning this repository and then

```
pip install -e .
```

## Quick Start

```python
import torch
from dpy5 import DiffCanvas
# Create a canvas (width, height)
c = DiffCanvas(256, 256)
c.background(1.0)               # white background
c.fill(1.0, 0.0, 0.0, 1.0)      # red
c.stroke(0.0)    # black stroke
c.stroke_weight(2.0)
c.polyline([[50, 50], [200, 50], [200, 200], [50, 200]], close=True)

# Render the scene (differentiable, img contains the resulting tensor)
img = c.render() # Returns tensor
c.get_image()
``` 
The corresponding DiffVG scene is cleared when `background` is called (think of it as a `begin`) and then re-constructed each time as drawing commands are called. Calling `render` at the end rasterizes the scene while allowing gradient propagation to any parameter used in the drawing procedures.

A typical optimization loop, involves re-drawing the scene at each step and using the otuput of `render` to compute some loss function with respect to the rendered image.

### Optimization example
All drawing functions accept python sequences (e.g. lists, tuples, numpy arrays or pytorch tensors). Passing arguments as tensors with gradients enabled (`requires_grad=True`) will enable gradient propagation to the arguments. While doing so explictly is possible (e.g. `c.polyline(some_tensor)`), the API allows you to do so more concisely by using the `c.var(value, group_name)` syntax. 

This method returns a PyTorch tensor with `requires_grad=True` and providing a `group_name` caches the tensor so it can be retrieved later with `c.get_vars(group_name)` and passed on to an optimizer of choice. The values passed in to `c.var` will be used as initial values for the tensor, but in subsequent calls to the drawing sequence `DiffCanvas` will use the cached tensors instead of creating new ones, as long as the same variable creation order is maintained.

> **Note:** while this caching method saves typing, it expects the drawing order and tensor sizes to remain unchanged for each step of the optimization. 

Here is a simple example of an optimization loop that adapts a series of curves to minimize the L1 error with a target image and displays the results:

``` python
import os
import torch
import matplotlib.pyplot as plt
from dpy5 import DiffCanvas
from PIL import Image
from tqdm import tqdm
import numpy as np

target_img = Image.open('./spock256.jpg')
w, h = target_img.size

c = DiffCanvas(w, h)

# This function will be called repeatedly during optimization
# and the `c.var` variables will set on the first call to draw and then
# change during optimization
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
plt.show()

```

![result](./examples/image_example_02.png)


## API Overview

### Canvas Setup

```python
canvas = DiffCanvas(width, height, device=None)
```

- `width`, `height`: image size in pixels.
- `device`: PyTorch device (defaults to CUDA if available).

### Rendering

```python
canvas.render(prefiltering=False, num_samples=2, seed=0, sdf=False)
```

- `prefiltering`: if `True`, uses an anti‑aliasing prefilter. Produces crisper lines, but does not support variable width strokes and produces artefacts in some cases.
- `num_samples`: multisampling level.
- `sdf`: if `True`, outputs a signed distance field.

After rendering, the result is stored in `canvas.img`. Retrieve it as a PIL image with `canvas.get_image()` or as a NumPy array with `canvas.get_array()`.

### Drawing State

| Method | Description |
|--------|-------------|
| `fill(*args)` | Set fill color. Accepts 1-4 numbers/tensors. |
| `stroke(*args)` | Set stroke color. |
| `stroke_weight(w)` | Set line width. |
| `push()` / `pop()` | Save/restore transformation and style. |
| `push_matrix()` / `pop_matrix()` | Save/restore only the transformation matrix. |
| `push_style()` / `pop_style()` | Save/restore only style attributes. |
| `translate(x, y)` | Apply translation. |
| `rotate(angle)` | Apply rotation (in radians by default). |
| `scale(sx, sy)` | Apply scaling. |
| `identity()` / `reset_matrix()` | Reset the current transformation to identity. |
| `angle_mode(mode)` | Set angle mode: `'radians'` or `'degrees'`. |
| `rect_mode(mode)` | (future) Set rectangle drawing mode. |
| `ellipse_mode(mode)` | (future) Set ellipse drawing mode. |
| `fill_rule(rule)` | Set fill rule: `'evenodd'`, `'nonzero'`, `'winding'`. |
| `curve_tightness(val)` | Set tension for cardinal splines (0‑1, default 0.5). |

### Drawing Primitives

| Method | Description |
|--------|-------------|
| `line(x0, y0, x1, y1)` or `line([x0,y0], [x1,y1])` | Draw a straight line. |
| `polyline(points, close=False)` | Draw an open or closed polyline. |
| `multibezier(points, close=False)` | Draw a sequence of cubic Bézier segments. |
| `curve(points, close=False)` | Draw a smooth cardinal spline through the given points. |
| `shape(obj, close=False)` | Draw a `Shape` object or a list of polylines. |
| `rect(x, y, w, h, radius=None)` or `rectangle(...)` | Draw a rectangle. Optional `radius` for rounded corners. |
| `square(x, y, size)` | Draw a square. |
| `ellipse(x, y, w, h)` | Draw an ellipse. |
| `circle(x, y, r)` | Draw a circle. |
| `triangle(a, b, c)` | Draw a triangle from three points. |
| `quad(a, b, c, d)` | Draw a quadrilateral from four points. |

### Complex Shapes

Build shapes piece by piece, similar to Processing’s `beginShape()` / `endShape()`:

```python
canvas.begin_shape()
canvas.begin_contour()
canvas.vertex(0, 0)
canvas.bezier_vertex(100, 50, 200, 50, 300, 0)
canvas.end_contour()

canvas.begin_contour()
canvas.curve_vertex(0, 200)
canvas.curve_vertex(100, 150)
canvas.curve_vertex(200, 150)
canvas.curve_vertex(300, 200)
canvas.end_contour(close=True)

canvas.end_shape()
```

The `Shape` class can also be used standalone:

```python
s = Shape(tension=0.5)
s.begin_shape()
s.vertex(...)
s.end_shape()
canvas.shape(s)
```

Calling `canvas.shape(s)` multiple times will instance the same geometry with the current transformation, reusing the underlying `pydiffvg` paths.


