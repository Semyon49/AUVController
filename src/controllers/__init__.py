from .motion import MotionController
from .depth import DepthController
from .auv import DrivingController

__all__ = [
    "MotionController", # Контроллер движения
    "DepthController",   # Контроллер глубины
    "DrivingController"               # Главный контроллер AUV
]
