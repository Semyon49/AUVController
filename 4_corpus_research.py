import cv2
import numpy as np
import pymurapi as mur
from typing import Tuple, List


# --- Конфигурация ---
class Config:
    """Конфигурация параметров."""
    AUV = mur.mur_init()
    IMAGE_SHAPE = AUV.get_image_bottom.shape

    SPEED = 20
    K_SPEED = 0.1
    LOWER_BOUND_WHITE = np.array([0, 0, 190])
    UPPER_BOUND_WHITE = np.array([360, 20, 255])
    LOWER_BOUND_ORANGE = np.array([10, 100, 100])
    UPPER_BOUND_ORANGE = np.array([35, 255, 255])
    # TARGET_DEPTH = 3.4
    AREA_THRESHOLD = 53000

    KP = 1.7
    KI = 0.4
    KD = 1


# --- Интерфейсы ---
class ImageProcessorInterface:
    """Интерфейс обработки изображений."""
    @staticmethod
    def get_contours(self, image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> List[np.ndarray]:
        """Создаёт маску на основе границ HSV."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def find_extreme_points(self, contours: List[np.ndarray]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Находит крайние точки у контура."""
        all_points = np.vstack(contours).squeeze()
        leftmost = all_points[np.argmin(all_points[:, 0])]
        rightmost = all_points[np.argmax(all_points[:, 0])]
        topmost = all_points[np.argmin(all_points[:, 1])]
        bottommost = all_points[np.argmax(all_points[:, 1])]

        pt1 = (leftmost[0], topmost[1])
        pt2 = (rightmost[0], bottommost[1])
        return pt1, pt2

    @staticmethod
    def calculate_area(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> int:
        """Вычисляет площадь прямоугольника."""
        return (pt2[0] - pt1[0]) * (pt2[1] - pt1[1])

# --- Реализация обработки изображений ---
class ImageProcessor(ImageProcessorInterface):
    ...

# --- Управление глубиной (PID-регулятор) ---
class DepthController:
    """Класс для управления глубиной AUV."""
    def __init__(self, settings: Config = Config, target_depth: float = 0) -> None:
        self.kp = settings.KP
        self.ki = settings.KI
        self.kd = settings.KD
        self.target_depth = target_depth
        self.integral = 0
        self.prev_error = 0
        self.last_time = None
        # print('Init DepthController is good')

    def __calculate_power(self, current_depth: float) -> float:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        delta_time = (current_time - self.last_time) if self.last_time else 0
        self.last_time = current_time

        error = self.target_depth - current_depth
        proportional = self.kp * error
        self.integral += error * delta_time
        integral = self.ki * self.integral
        derivative = self.kd * ((error - self.prev_error) / delta_time) if delta_time > 0 else 0
        self.prev_error = error

        return proportional + integral + derivative
    
    def set_target_depth(self, target_depth: float) -> None:
        self.target_depth = target_depth

    def control_depth(self, current_depth: float) -> float:
        return self.__calculate_power(current_depth)


# --- Управление двигателями ---
class MotionController:
    """Класс для управления двигателями."""
    def __init__(self, config: Config = Config) -> None:
        self.auv = config.AUV
        self.image_shape = config.IMAGE_SHAPE
        self.speed: int = config.SPEED
        self.k: float = config.K_SPEED

    def adjust_power(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> None:
        mtr_power_l = pt1[0]
        mtr_power_r = self.image_shape[1] - pt2[0]
        power_l = self.speed + mtr_power_l * self.k
        power_r = self.speed + mtr_power_r * self.k

        self.auv.set_motor_power(0, power_l)
        self.auv.set_motor_power(1, power_r)

    def swim_up(self, speed: int) -> None:
        self.auv.set_motor_power(2, speed)
        self.auv.set_motor_power(3, speed)

    def stop(self) -> None:
        for i in range(5):
            self.auv.set_motor_power(i, 0)

# --- Логика навигации ---
class AUVNavigator:
    ...

class AUVController(MotionController, DepthController, AUVNavigator):
    def __init__(self, config: Config) -> None:
        MotionController.__init__(config)
        DepthController.__init__(config)
        AUVNavigator.__init__()
        self.config = config
        print('Init AUVController is good')
        

class ABC:
    def __init__(self, draw: bool = False):
        self.auv = mur.mur_init()
        self.config = Config()
        self.navigator = AUVController(self.motor_controller, self.depth_controller, self.config)


# --- Запуск ---
if __name__ == "__main__":
    app = AUVController(Config)
    app.set_target_depth(2)