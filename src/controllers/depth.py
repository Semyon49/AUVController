import cv2
import time
import numpy as np
import pymurapi as mur

from settings import Config

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

        return - (proportional + integral + derivative)

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