# py5diff - Processing-like Differentiable Vector Graphics

`py5diff` provides a Processing‑inspired API (e.g., `push()`, `pop()`, `fill()`, `stroke()`, `line()`, `curve()`) for building **differentiable* 2D vector scenes. Under the hood it uses torch and [pydiffvg](https://github.com/BachiLi/diffvg) and PyTorch, so all parameters are tensors, and gradients can flow through the rendering process.

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
from py5diff import DiffCanvas
# Create a canvas (width, height)
c = DiffCanvas(256, 256)
c.background(1.0)               # white background
c.fill(1.0, 0.0, 0.0, 1.0)      # red
c.stroke(0.0)    # black stroke
c.stroke_weight(2.0)
c.polyline([[50, 50], [200, 50], [200, 200], [50, 200]], close=True)

# Render the scene (differentiable, img contains the resulting tensor)
img = canvas.render()
c.get_image()

``` 

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

- `prefiltering`: if `True`, uses an anti‑aliasing prefilter.
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


