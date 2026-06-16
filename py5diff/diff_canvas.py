#!/usr/bin/env python3
import numbers
import torch
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
import pydiffvg
import copy
from PIL import Image

def make_mat(M, device, dtype):
    return torch.vstack([torch.stack([torch.as_tensor(v) for v in row]) for row in M]).to(dtype).to(device)

def make_vec(vec, device, dtype):
    return torch.stack([torch.as_tensor(v) for v in vec])

def is_number(x):
    return isinstance(x, numbers.Number)
    
class CanvasState:
    ''' Keeps track of styles etc to enable push/pop'''
    def __init__(self, c):
        self.c = c
        self.cur_fill = c._get_color(1.0)
        self.cur_stroke = c._get_color(0.0)
        self._rect_mode = "corner"
        self._ellipse_mode = "center"
        self._line_width = 1.0
        self._angle_mode = 'radians'
        self._tension = 0.5
        self._fill_rule = 'evenodd'
         
    def set(self, prev=None):
        ''' Called if pop_style/pop is called in canvas'''
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
    "_tension"
)


class DiffCanvas:
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
        self.building = False

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
        
    def begin(self):
        @contextmanager
        def popmanager():
            pass
            try:
                yield
            finally:
                self.end()
        self.reset()
        self.building = True
        return popmanager()

    def end(self):
        self.building = False
        
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

    def _vec(self, v):
        return make_vec(v, self.device, self.dtype)
    
    def _to(self, v):
        return torch.as_tensor(v).to(self.dtype).to(self.device)

    def translate(self, *args):
        if len(args) == 1:
            p = self._to(args[0])
            x, y = p
        else:
            x, y = [self._to(v) for v in args]
        M = self._mat([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])
        self._transform = self._transform @ M
        
    def rotate(self, angle):
        # angle in rad, differentiable
        angle = self._to(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        M = self._mat([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self._transform = self._transform @ M

    def scale(self, *args):
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
        self._tranform = torch.tensor(np.eye(3), device=self.device, dtype=self.dtype)

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
            return torch.as_tensor(self.cur_stroke) #* self.color_scale
        if self.cur_fill is not None:
            return torch.as_tensor(self.cur_fill) #* self.color_scale
        return None

    @property
    def center(self):
        """The center of the canvas (as a 2d numpy array)"""
        return self._to([self._width / 2,
                         self._height / 2])

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

    def no_stroke(self):
        """Do not stroke subsequent shapes"""
        self.stroke(None)

    def fill_rule(self, rule):
        """Sets the fill rule for complex shapes.

        Arguments:
        - One of `"evenodd"`, `"nonzero"`, or `"winding"`
        """
        self._fill_rule = rule
    
    def angle_mode(self, mode='degrees'):
        mode = mode.lower()
        if not mode in ['degrees', 'radians']:
            raise ValueError('invalid angle mode, use either RADIANS or DEGREES')
        self._angle_mode = mode

    def _to_radians(self, ang):
        if self._angle_mode == 'radians':
            return ang
        return radians(ang)

    def _to_degrees(self, ang):
        if self._angle_mode == 'degrees':
            return ang
        return degrees(ang)

    def _get_color(self, *args):
        if len(args) == 1:
            if not is_number(args[0]):
                x = args[0]
                if len(x) == 4:
                    return self._to(x)
                elif len(x) == 3:
                    return torch.cat([self._to(x), self._vec([1.0])])
                elif len(x) == 2:
                    return self._vec([x[0], x[0], x[0], x[1]])
                else:
                    return self._vec([x[0], x[0], x[0], 1.0])
            else:
                return self._vec([args[0], args[0], args[0], 1.0])
        elif len(args) == 2:
            return self._vec([args[0], args[0], args[0], args[1]])
        elif len(args) == 3:
            return self._vec([args[0], args[1], args[2], 1.0])
        elif len(args) == 4:
            return self._vec(args)
        raise ValueError("Invalid arg combination")

    def background(self, *args):
        """Clear the canvas with a given color
        Accepts either a tensor with the color components, or single color components (as in `fill`)
        Currently no alpha
        """
        
        if not len(args):
            raise ValueError("background requires at least one argument")

        # Background clears so we may as well begin
        self.begin()
        
        if args[0] is None:
            self._bg = None
            return

        clr = self._get_color(*args)[:3] 
        self._bg = torch.zeros(self.height, self.width, 3, dtype=torch.float32, device=self.device)
        self._bg[...] = torch.as_tensor(clr).to(self.dtype).to(self.device)
        
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
            self.cur_stroke = self._get_color(args)

    def stroke_weight(self, w):
        """Set the line width

        Arguments:
        - The width in pixel of the stroke
        """
        self._line_width = w
        
    def curve_tightness(self, val):
        """Sets the 'tension' parameter for the curve used when using `curve_vertex`"""
        self._tension = val
        if self.cur_shape is not None:
            self.cur_shape.tension = val

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

        
    def begin_shape(self):
        """Start building a complex shape. Drawing is deferred until end_shape()."""
        self.cur_shape = Shape(tension=self._tension)
        self.cur_shape.begin_shape()
        self.building_shape = True

    def end_shape(self, close=False):
        """Finish the shape and draw it."""
        if self.cur_shape is None:
            return
        self.building_shape = False
        self.cur_shape.end_shape(close)
        self._build_shape(self.cur_shape)
        self.cur_shape = None
        
    def begin_contour(self):
        """Start a new contour within the currently built shape.
        If no shape is active, a new one is created automatically."""
        if self.cur_shape is None:
            self.cur_shape = Shape(tension=self._tension)
            self.cur_shape.begin_shape()
        self.cur_shape.begin_contour()

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

    def vertex(self, *args):
        """Add a vertex to current contour

        Input arguments can be in the following formats:

        - `[x, y]`
        - `x, y`
        """
        if self.cur_shape is None:
            raise RuntimeError("vertex() called without begin_shape()")
        self.cur_shape.vertex(*args)

    def curve_vertex(self, *args):
        """Add a curved vertex to current contour

        Input arguments can be in the following formats:

        - `[x, y]`
        - `x, y`
        """
        if self.cur_shape is None:
            raise RuntimeError("curve_vertex() called without begin_shape()")
        self.cur_shape.curve_vertex(*args)

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

    def polyline(self, *args, close=False):
        """Draw a polyline (open by default).

        The polyline is specified as either:

        - a list of =[x,y]= pairs (e.g. =[[0, 100], [200, 100], [200, 200]]=)
        - a tensor array with shape =(n, 2)=, representing =n= points (a point for each row and a coordinate for each column)
        - two 1d sequences/tensors, one for each coordinate

        To close the polyline set the named =close= argument to =True=, e.g. =c.polyline(points, close=True)=.
        """
        if len(args) == 1:
            points = args[0]
        elif len(args) == 2:
            points = torch.vstack(args).T
        else:
            raise ValueError("Wrong number of arguments")
        self.begin_contour()
        self.cur_shape._polyline(points, close)
        self.end_contour(close)

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
            points = args[0]
        elif len(args) == 2:
            points = torch.vstack(args).T
        else:
            raise ValueError("Wrong number of arguments")
        self.begin_contour()
        self.cur_shape._multibezier(points, close)
        self.end_contour(close)

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
            points = args[0]
        elif len(args) == 2:
            points = torch.vstack(args).T
        else:
            raise ValueError("Wrong number of arguments")
        self.begin_contour()
        self.cur_shape._curve(points, close)
        self.end_contour(close)

    def shape(self, obj, close=False):
        """Draw a pre‑built Shape object or a list of polylines (list of lists/arrays).
        For lists, each polyline becomes one contour (open or closed)."""
        
        if isinstance(obj, Shape):
            if obj in self.shape_to_inds[obj]:
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

    ###############################################
    # Scene management

    def _add_primitives(self, primitives):
        ''' Add new primitives for rendering'''
        ind = len(self.primitives)
        self.primitives += primitives
        shape_ids = list(range(ind, ind+len(primitives)))
        self._instance_primitives(shape_ids)

    def _instance_primitives(self, shape_ids):
        ''' Create groups for given primitive indices'''
        fill_color = None
        if self.cur_fill is not None:
            fill_color = self.cur_fill.to(self.device)
        stroke_color = None
        if self.cur_stroke is not None:
            stroke_color = self.cur_stroke.to(self.device)
                
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor(shape_ids),
                                    use_even_odd_rule=self._fill_rule=='evenodd',
                                    fill_color=fill_color,
                                    stroke_color=stroke_color)
        group.shape_to_canvas = self._transform.to(self.device)
        self.groups.append(group)
        
    def _build_shape(self, shape):
        primitives = shape.build(self)
        inds = self._add_primitives(primitives)
        # store for instancing if shape is called with same object multiple times
        self.shape_to_inds[shape] = inds
        
    # Override simple shapes
    def line(self, *args):
        if len(args) == 2:
            a, b = [torch.as_tensor(v) for v in args]
        elif len(args) == 4:
            ax, ay, bx, by = [torch.as_tensor(v) for v in args]
            a = torch.stack([ax, ay])
            b = torch.stack([bx, by])
        else:
            raise ValueError("line: Unexpected number of arguments")

        self.polyline(torch.vstack([a,b]))

        
    # def circle(self, *args, mode=None):
    #     def geom(t):
    #         c = t @ center
    #         r_t = r if r.requires_grad else r  # handle scaling correctly…
    #         return [self._circle_path(c, r_t)]
    #     self._add_item(geom)
    

    # Shape building (same as parent, but end_shape appends)
    def render(self, prefiltering=False, num_samples=2, seed=0, sdf=False):       
        if prefiltering:
            num_samples = 1

        if self._bg is not None:
            bg = torch.as_tensor(self._bg).to(self.dtype).to(self.device)
            if len(bg.shape)==2:
                bg = bg[:,:, np.newaxis]
                bg = bg.repeat(1, 1, 3)
            h, w, _ = bg.shape
        else:
            w, h = self.width, self.height

        if not self.primitives:
            self.img = bg
            return
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h,
                                                             self.primitives,
                                                             self.groups,
                                                             use_prefiltering=prefiltering,
                                                             output_type=pydiffvg.OutputType.sdf if sdf
                                                             else pydiffvg.OutputType.color)
        try:
            img = pydiffvg.RenderFunction.apply(w, h, num_samples, num_samples, seed, None, *scene_args)
        except RuntimeError as e:
            print("RUNTIME ERROR IN RENDER")
            print("Possibly wrong dtype in geometry, needs to be float32")
            raise(e)

        if self._bg is not None:
            img = img[:, :, 3:4] * img[:, :, :3] + bg * (1 - img[:, :, 3:4])
            img = img[:, :, :3]

        self.img = img
        
    def get_image(self):
        assert self.img is not None
        img = self.img.detach().cpu().numpy()
        return Image.fromarray((img*255).astype(np.uint8))
    
    def get_array(self):
        assert self.img is not None
        img = self.img.detach().cpu().numpy()
        return img

    def _repr_png_(self):
        """Tells Jupyter to render this object as a PNG image."""
        import io
        byte_arr = io.BytesIO()
        self.get_image().save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    ## Variable management
    def var(self, v, group_name='', grad=True, id=None):
        '''
        Return a tensor for the given variable, input can be a tensor or a sequence

        If `group_name` is provided, the variable is cached, allowing it to be reused across
        multiple drawing operations without recreating the tensor, e.g
        ```python
        c.curve(c.var([[0,0], [100,0], [100,100]], 'pts'))
        c.curve(c.var([[20,0], [10,40], [100,100]], 'pts'))
        ```
        Will cache two tensors that can be retrieved as a list with `c.get_vars('pts')`.
        Modifying these tensors will modify the values used in subsequent calls to the
        same drawing sequence, meaning we can optimize the variable in a loop.
        While handy, note that changing the rendering order after these variables are cached,
        will result in unexpected behaviors.
        '''
        v = torch.as_tensor(v).to(self.dtype).to(self.device)
        v.requires_grad = grad
        if group_name:
            if id is None:
                if group_name in self._vars:
                    id = len(self._vars[group_name])
                else:
                    id = 0
            var_id = self._var_id(group_name, id)
            if var_id in self._id_to_var:
                return self._id_to_var[var_id]
            self._vars[group_name].append(v)
            self._id_to_var[var_id] = v
            
        return v

    def get_vars(self, group_name):
        if group_name not in self._vars:
            return []
        return self._vars[group_name]

    def clear_vars(self):
        self._vars = defaultdict(list)
        self._id_to_var = {}

    def _var_id(self, name, id):
        return f'{name}_{id}'

    
def num_bezier(n_ctrl, closed=False, degree=3):
    if not is_number(n_ctrl):
        n_ctrl = len(n_ctrl)
    if closed:
        n_ctrl += 1
    return int((n_ctrl - 1) / degree)


class Shape:
    """
    Holds a list of contours, each contour being a sequence of drawing commands.
    Mirrors Processing's PShape: use begin_shape()/end_shape() to construct.
    """
    def __init__(self, tension=0.5):
        self.tension = tension
        self.contours = []           # list of contour command lists
        self.reset()
        
    def reset(self):
        self._contour = [] # list of commands for the contour being built
        self._num_ctrl = []
        self._curve_points = []      # pending Catmull‑Rom points for curve_vertex
        self._spline_start = None    # first point of the current spline (move-to)
        self._shape_active = False         # True between begin_shape()/end_shape()

    # --- Public construction methods (mirror PShape) ---
    def begin_shape(self):
        """Start building the shape. Clears any previous geometry."""
        self.reset()
        self._shape_active = True
        self.contours = []
        
    def end_shape(self, close=False):
        """
        Finish building the shape.
        If close is True, the last contour is closed before finalising.
        """
        if self._shape_active:
            if self._current or self._curve_points:
                self.end_contour(close)
            self._shape_active = False

    def begin_contour(self):
        """Start a new contour. Must be called after begin_shape()."""
        if not self._shape_active:
            raise RuntimeError("begin_shape() must be called before begin_contour()")
        self.reset()
        
    def end_contour(self, close=False):
        """Finish the current contour. If close=True, the contour is closed."""
        self._flush_spline(close=close)
        pts = torch.vstack(self._contour)
        n_ctrl = torch.tensor(self._num_ctrl, dtype=torch.int32)
        self.contours.append((pts, n_ctrl, close))
        self.reset()
        
    def vertex(self, *args):
        """Add a straight vertex ."""
        if len(args) > 1:
            x = torch.stack([torch.as_tensor(v) for v in args])
        else:
            x = args[0]
        self._start_contour_if_needed()
        self._flush_spline()
        self._contour.append(x)
        self._num_ctrl += [1] 
        
    def curve_vertex(self, x, y=None):
        """Add a curved vertex (Catmull Rom spline)."""
        if len(args) > 1:
            x = torch.stack([torch.as_tensor(v) for v in args])
        else:
            x = args[0]
            
        self._start_contour_if_needed()
        if not self._curve_points:
            if self._contour and self._num_ctrl[-1] == 0:
                self._spline_start = self._contour[-1][-1].clone()
            else:
                self._spline_start = None
        self._curve_points.append(x)

    def bezier_vertex(self, *args):
        """Add a cubic Bézier vertex; three control points."""
        
        pts = torch.vstack(args)
        self._start_contour_if_needed()
        self._flush_spline()
        self._contour.append(pts)
        self._num_ctrl += [2] 

    def _polyline(self, points, closed):
        self._contour.append(torch.as_tensor(points))
        self._num_ctrl += [0]*len(points)
        
    def polyline(self, points, close=False):
        """Add a contour of straight line segments from a sequence of (x,y) points."""
        if not self._shape_active:
            self.begin_shape()          # temporary activation for standalone use
        self.begin_contour()
        self._polyline(points, close)
        self.end_contour(close)

    def _multibezier(self, points, close):
        num_segs = num_bezier(points, close)
        self._contour.append(torch.as_tensor(points))
        self._num_ctrl += [2]*num_segs
        
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
        self._multibezier(cardinal_spline(torch.as_tensor(points), self.tension, close), close)
                 
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
        if not self._current:
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
        num_segs = num_bezier(cp, closed)
        self._contour.append(torch.as_tensor(cp))
        self._num_ctrl += [2]*num_segs
        
        self._curve_points = []
        self._spline_start = None

    def build(self, c):
        """Build diffvg primitives"""
        shapes = []
        for ctr in self.contours:
            if isinstance(ctr, tuple):
                pts, nctrl, closed = ctr
                if pts.shape[1] > 2:
                    w = pts[:,2].to(c.device)
                else:
                    w = torch.as_tensor(c._line_width).to(c.device)
                path = pydiffvg.Path(num_control_points=nctrl.to(c.device),
                                    points=pts[:,:2].to(c.dtype).to(c.device),
                                    stroke_width = w,
                                    is_closed=closed)
                shapes.append(path)
            else: # Assume a diffVg object added externally by canvas
                shapes.append(ctr)
        return shapes


def cardinal_spline(Q, c, closed=False):
    ''' Cardinal spline interpolation for a sequence of values'''
    isnp = isinstance(Q, np.ndarray)

    if closed:
        if isnp:
            Q = np.vstack([Q, Q[0:1]])
        else:
            Q = torch.concat([Q, Q[0:1]])
    n = len(Q)
    D = []
    for k in range(1, n-1):
        # Assuming uniform parametrisation here
        d = (1-c)*(Q[k+1] - Q[k-1])
        D.append(d)
    if closed:
        d1 = dn = (1-c)*(Q[1] - Q[-2])
    else:
        d1 = (1-c)*(Q[1] - Q[0])
        dn = (1-c)*(Q[-1] - Q[-2])
    D = [d1] + D + [dn]
    P = [Q[0]]
    for k in range(1, n):
        p1 = Q[k-1] + D[k-1]/3
        p2 = Q[k] - D[k]/3
        p3 = Q[k]
        P += [p1, p2, p3]

    if closed:
        P = P[:-1]
    if isnp:
        return np.vstack(P)
    return torch.vstack(P)


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
    if (isinstance(S[0], torch.Tensor) or isinstance(S[0], np.ndarray)) and len(S[0].shape) > 1:
        return True
    return False

def default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        # DiffVG does not work well with ARM
        return torch.device('cpu')
        
