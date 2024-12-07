import cv2
import numpy as np

# --- Интерфейсы ---
class ImageProcessorInterface:
    """Interface for image processing."""

    @staticmethod
    def get_contours(image, lower_bound = None, upper_bound = None): #-> list[np.ndarray]
        """
        Creates a mask based on HSV boundaries.

        Args:
            image (np.ndarray): Input image.
            lower_bound (np.ndarray): Lower HSV boundary.
            upper_bound (np.ndarray): Upper HSV boundary.

        Returns:
            list[np.ndarray]: List of contours found in the mask.
        """
        if lower_bound == None and upper_bound == None:
            return self.__get_contours_bin(image)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
        
    @staticmethod
    def __get_contours_bin(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
                return contours[0] 

    @staticmethod
    def find_extreme_points(contours): # -> tuple[tuple[int, int], tuple[int, int]]
        """
        Finds the extreme points of a contour.

        Args:
            contours (list[np.ndarray]): List of contours.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: Top-left and bottom-right points of the rectangle.
        """
        all_points = np.vstack(contours).squeeze()
        print(all_points)
        if len(all_points) >= 4:
            leftmost = all_points[np.argmin(all_points[:, 0])]
            rightmost = all_points[np.argmax(all_points[:, 0])]
            topmost = all_points[np.argmin(all_points[:, 1])]
            bottommost = all_points[np.argmax(all_points[:, 1])]

            pt1 = (leftmost[0], topmost[1])
            pt2 = (rightmost[0], bottommost[1])
            return pt1, pt2
        return (0, 0), (0, 0)

        all_points = np.vstack(contours).squeeze()
        leftmost = all_points[np.argmin(all_points[:, 0])]
        rightmost = all_points[np.argmax(all_points[:, 0])]
        topmost = all_points[np.argmin(all_points[:, 1])]
        bottommost = all_points[np.argmax(all_points[:, 1])]

        pt1 = (leftmost[0], topmost[1])
        pt2 = (rightmost[0], bottommost[1])
        return pt1, pt2

    @staticmethod
    def calculate_area(pt1, pt2): # -> int
        """
        Calculates the area of a rectangle.

        Args:
            pt1 (tuple[int, int]): Top-left corner of the rectangle.
            pt2 (tuple[int, int]): Bottom-right corner of the rectangle.

        Returns:
            int: Area of the rectangle.
        """
        return (pt2[0] - pt1[0]) * (pt2[1] - pt1[1])