import cv2
import sys
import time
import numpy as np
import pymurapi as mur


# --- Конфигурация ---
class Config:
    """Configuration of parameters."""
    
    AUV = mur.mur_init()

    IMAGE_SHAPE = AUV.get_image_bottom().shape

    SPEED = 20
    K_SPEED = 0.1

    LOWER_BOUND_WHITE = np.array([0, 0, 190])
    UPPER_BOUND_WHITE = np.array([360, 20, 255])

    LOWER_BOUND_ORANGE = np.array([10, 100, 100])
    UPPER_BOUND_ORANGE = np.array([35, 255, 255])

    AREA_THRESHOLD = 53000

    KP = 1.7
    KI = 0.4
    KD = 1.0

    DRAW = True


# --- Интерфейсы ---
class ImageProcessorInterface:
    """Interface for image processing."""

    @staticmethod
    def get_contours(image, lower_bound, upper_bound): #-> list[np.ndarray]
        """
        Creates a mask based on HSV boundaries.

        Args:
            image (np.ndarray): Input image.
            lower_bound (np.ndarray): Lower HSV boundary.
            upper_bound (np.ndarray): Upper HSV boundary.

        Returns:
            list[np.ndarray]: List of contours found in the mask.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

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

        if contours:
            pt1, pt2 = ImageProcessorInterface.find_extreme_points(contours)

            if Config.DRAW:
                cv2.rectangle(image, pt1, pt2, (0, 255, 255), 5)
            return pt1, pt2

        return (0, 0), (0, 0)

# --- Управление глубиной (PID-регулятор) --- 
class DepthController:
    """Class for managing the depth of an AUV (Autonomous Underwater Vehicle)."""

    def __init__(self, settings=Config, target_depth=0): # -> None
        """
        Initializes the depth controller.

        Args:
            settings (Config): Configuration object containing PID values.
            target_depth (float): Desired depth to reach.
        """
        self.kp = settings.KP
        self.ki = settings.KI
        self.kd = settings.KD
        self.target_depth = target_depth
        self.integral = 0
        self.prev_error = 0
        self.last_time = None

    def __calculate_power(self, current_depth): # -> float
        """
        Calculates the required power to reach the target depth.

        Args:
            current_depth (float): Current depth of the AUV.

        Returns:
            float: Calculated power based on PID control.
        """
        current_time = time.time()
        
        if self.last_time is not None:
            delta_time = current_time - self.last_time
        else:
            delta_time = 0
        
        self.last_time = current_time

        error = self.target_depth - current_depth
        proportional = self.kp * error
        self.integral += error * delta_time
        integral = self.ki * self.integral
        derivative = self.kd * ((error - self.prev_error) / delta_time) if delta_time > 0 else 0
        self.prev_error = error

        return proportional + integral + derivative

    def set_target_depth(self, target_depth): # -> None
        """
        Sets a new target depth.

        Args:
            target_depth (float): The new target depth.
        """
        self.target_depth = target_depth

    def control_depth(self, current_depth): # -> float
        """
        Controls the depth of the AUV.

        Args:
            current_depth (float): Current depth of the AUV.

        Returns:
            float: Calculated power to adjust the depth.
        """
        return self.__calculate_power(current_depth)



# --- Управление двигателями ---
class MotionController:
    """Class for controlling the motors of the AUV (Autonomous Underwater Vehicle)."""

    def __init__(self, config = Config): # -> None
        """
        Initializes the motion controller.

        Args:
            config (Config): Configuration object containing motor settings.
        """
        self.auv = config.AUV
        self.config = config
        self.image_shape = config.IMAGE_SHAPE
        self.speed = config.SPEED
        self.k = config.K_SPEED

    def adjust_power(self, pt1, pt2): # -> None
        """
        Adjusts the motor power based on the detected positions of the rectangle.

        Args:
            pt1 (tuple[int, int]): Top-left corner of the rectangle.
            pt2 (tuple[int, int]): Bottom-right corner of the rectangle.
        """
        mtr_power_l = pt1[0]
        mtr_power_r = self.config.IMAGE_SHAPE[1] - pt2[0]
        power_l = self.config.SPEED + mtr_power_l * self.config.K_SPEED
        power_r = self.config.SPEED + mtr_power_r * self.config.K_SPEED

        self.config.AUV.set_motor_power(0, power_l)
        self.config.AUV.set_motor_power(1, power_r)

    def set_power_swim_up(self, power): # -> None
        """
        Makes the AUV swim upwards by setting motor power.

        Args:
            power (int): The power to set for the upward movement.
        """
        self.config.AUV.set_motor_power(2, power)
        self.config.AUV.set_motor_power(3, power)

    def set_power_swim_forward(self, power): # -> None
        """
        Sets the power for both motors to the same value.

        Args:
            power (int): The power to set for both motors.
        """
        self.config.AUV.set_motor_power(0, power)
        self.config.AUV.set_motor_power(1, power)

    def stop(self): # -> None:
        """
        Stops all motors of the AUV.
        """
        for i in range(5):
            self.auv.set_motor_power(i, 0)


# --- Логика навигации ---
class AUVNavigator(ImageProcessor):
    ...

class AUVController(MotionController, DepthController, AUVNavigator):
    """Class for controlling the AUV (Autonomous Underwater Vehicle)."""
    
    def __init__(self, config = Config): # -> None
        """
        Initializes the AUV controller by setting up the motion, depth, and navigation controllers.
        
        Args:
            config (Config): Configuration object containing AUV settings.
        """
        MotionController.__init__(self, config)
        DepthController.__init__(self, config)
        self.config = config

    def corpus_research(self): # -> None
        """ Task № 4
        Conducts the corpus research by controlling the AUV to search for an orange arrow 
        and then navigate towards a rectangle.
        """
        # Searching for the orange arrow
        while True:
            image = self.config.AUV.get_image_bottom()
            pt1, pt2 = self.detection_figures(image, self.config.LOWER_BOUND_ORANGE, self.config.UPPER_BOUND_ORANGE)
            area = ImageProcessorInterface.calculate_area(pt1, pt2)
            
            if area > self.config.AREA_THRESHOLD:
                break

            if self.config.DRAW:
                cv2.imshow("Processed Image", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Navigating towards the white rectangle
        while True:
            image = self.config.AUV.get_image_front()
            pt1, pt2 = self.detection_figures(image, self.config.LOWER_BOUND_WHITE, self.config.UPPER_BOUND_WHITE)
            area = ImageProcessorInterface.calculate_area(pt1, pt2)

            if area == 0:
                self.set_power_swim_forward(20)
                continue
            elif abs(53000 - area) < 2000:
                break

            self.adjust_power(pt1, pt2)

            if self.config.DRAW:
                cv2.imshow('Processed Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def ascent(self) -> None:
        """ Task № 5
        Makes the AUV ascend.
        """
        # Implement the ascent functionality here
        pass

    
# --- Запуск ---
if __name__ == "__main__":
    app = AUVController(Config)
    app.set_target_depth(3.4)
    app.corpus_research()