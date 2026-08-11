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
"""

from . import diff_canvas
from .diff_canvas import DiffCanvas, Shape
from .utils import CanvasOptimizer, MultiLoss, show_animation, perf_timer
from . import utils
