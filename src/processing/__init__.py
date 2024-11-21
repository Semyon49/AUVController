# processing/__init__.py

from .base_processor import ImageProcessor
from .interfaces import ImageProcessorInterface

__all__ = [
    "ImageProcessor",           # Класс обработки изображений
    "ImageProcessorInterface"   # Интерфейс обработки изображений
]
