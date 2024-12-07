import cv2
import numpy as np
import pymurapi as mur

from settings import Config
from .interfaces import ImageProcessorInterface

# --- Реализация обработки изображений ---
class ImageProcessor():
    """Image processing implementation."""

    def find_circles(self, image, lower_red1, upper_red1, lower_red2, upper_red2):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        red_objects = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(red_objects, cv2.COLOR_BGR2GRAY)

        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=20, maxRadius=100
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)  
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  

                cv2.imshow('Detected Red Circles', image)
                cv2.waitKey(1)

                return x, y
        return 0, 0
        

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
        
        if contours:
            pt1, pt2 = ImageProcessorInterface.find_extreme_points(contours)

            if Config.DRAW:
                cv2.drawContours(image, contours, -1, (255, 255, 0))
                cv2.imshow("winname", image)
                cv2.waitKey(1)

            return pt1, pt2

        return (0, 0), (0, 0)
