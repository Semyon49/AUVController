import cv2
import numpy as np
import pymurapi as mur

from settings import Config
    
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

    def adjust_motor_power(self, pt1, pt2): # -> None
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

    def adjust_motor_power_to_center(self, pt1):
        if pt1[1] > self.config.IMAGE_SHAPE[1]:
            power = 1
        else:
            power = -1

        mtr_power_l = self.config.IMAGE_SHAPE[0] - pt1[0]
        mtr_power_r = pt1[0] - self.config.IMAGE_SHAPE[0]

        power_l = self.config.SPEED * power + mtr_power_l * self.config.K_SPEED
        power_r = self.config.SPEED * power + mtr_power_r * self.config.K_SPEED

        self.config.AUV.set_motor_power(0, power_l)
        self.config.AUV.set_motor_power(1, power_r)

    def swim_up(self, power): # -> None
        """
        Makes the AUV swim upwards by setting motor power.

        Args:
            power (int): The power to set for the upward movement.
        """
        self.config.AUV.set_motor_power(2, power)
        self.config.AUV.set_motor_power(3, power)

    def swim_forward(self, power, power_right=None): 
        """
        Sets the power for both motors. If only one value is provided, both motors 
        are set to the same power. If two values are provided, they are applied 
        separately.

        Args:
            power (int): The power to set for the left motor, or both if power_right is None.
            power_right (int, optional): The power to set for the right motor. Defaults to None.
        """
        if power_right is None:
            power_right = power
        self.config.AUV.set_motor_power(0, power)
        self.config.AUV.set_motor_power(1, power_right)

    def stop_all_motors(self): # -> None:
        """
        Stops all motors of the AUV.
        """
        for i in range(5):
            self.auv.set_motor_power(i, 0)