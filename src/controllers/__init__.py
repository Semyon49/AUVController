# controllers/__init__.py

from .motion import MotionController
from .depth import DepthController
from .auv import AUVController

__all__ = [
    "MotionController",  # Контроллер движения
    "DepthController",   # Контроллер глубины
    "AUVController"      # Главный контроллер AUV
]
