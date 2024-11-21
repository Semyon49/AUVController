import cv2
import numpy as np
import pymurapi as mur

from settings import Config
from .interfaces import ImageProcessorInterface

# --- Реализация обработки изображений ---
class ImageProcessor():
    """Image processing implementation."""

    def detection_figures(self, image, lower_bound, upper_bound): # -> tuple[tuple[int, int], tuple[int, int]]
        """
        Follows the figures all color.

        Args:
            image (np.ndarray): Input image.
            lower_bound (): 
            upper_bound ():

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: Top-left and bottom-right points of the rectangle.
        """
        contours = ImageProcessorInterface.get_contours(image, lower_bound, upper_bound)

        cv2.drawContours(image, contours, -1, (255, 255, 0), 3)

        cv2.imshow('winname', image)
        cv2.waitKey(1)

        if contours:
            pt1, pt2 = ImageProcessorInterface.find_extreme_points(contours)

            if Config.DRAW:
                cv2.rectangle(image, pt1, pt2, (0, 255, 255), 5)
            return pt1, pt2

        return (0, 0), (0, 0)
