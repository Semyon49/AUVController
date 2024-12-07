import cv2
import time
import numpy as np

from settings import Config
from processing import ImageProcessor, ImageProcessorInterface
from controllers import MotionController, DepthController, DrivingController

class AUVController(MotionController, DepthController, DrivingController, ImageProcessor):
    """Class for controlling the AUV (Autonomous Underwater Vehicle)."""
    
    def __init__(self, config = Config): # -> None
        """
        Initializes the AUV controller by setting up the motion, depth, and navigation controllers.
        
        Args:
            config (Config): Configuration object containing AUV settings.
        """
        MotionController.__init__(self, config)
        DepthController.__init__(self, config)
        DrivingController.__init__(self, config)

        self.config = config

    def task1(self):
        self.folow_line()
        
    def corpus_research(self): # -> None
        """ Task № 4
        Conducts the corpus research by controlling the AUV to search for an orange arrow 
        and then navigate towards a rectangle.
        """
        # Searching for the orange arrow
        while True:
            current_depth = self.config.AUV.get_depth()
            power = self.control_depth(current_depth)
            self.swim_up(power)
            print(power)
            if abs(3.4 - current_depth) < 0.2:
                self.swim_up(-power)
                self.stop_all_motors()
                break

        while True:
            image = self.config.AUV.get_image_bottom()
            pt1, pt2 = self.detection_figures(image, self.config.LOWER_BOUND_ORANGE, self.config.UPPER_BOUND_ORANGE)
            area = ImageProcessorInterface.calculate_area(pt1, pt2)
            
            self.swim_forward(self.config.SPEED)
            print(pt1, pt2)
            if area > self.config.AREA_THRESHOLD:
                self.stop_all_motors()
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
                self.swim_forward(self.config.SPEED)
                continue
            elif abs(53000 - area) < 2000:
                break

            self.adjust_motor_power(pt1, pt2)

            if self.config.DRAW:
                cv2.imshow('Processed Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def ascent(self) -> None:
        """ Task № 5
        Makes the AUV ascend.
        """
        self.swim_forward(self.config.SPEED)
        time.sleep(1)
        self.stop_all_motors()
        self.set_target_depth(3.0)

        # while True:
        #     current_depth = self.config.AUV.get_depth()
        #     power = self.control_depth(current_depth)
        #     self.swim_up(power)

        #     if current_depth == 3:
        #         self.stop()
        #         return

        while True:
            image = Config.AUV.get_image_bottom()

            x , y = self.find_circles(image, app.config.LOWER_BOUND_RED1,app.config.UPPER_BOUND_RED1,app.config.LOWER_BOUND_RED2,app.config.UPPER_BOUND_RED2)
            if abs(160 - x) < 12:
                break
        
        self.set_target_depth(0)

        while True:
            current_depth = self.config.AUV.get_depth()
            power = self.control_depth(current_depth)
            self.swim_up(power)

            if current_depth == 0:
                self.stop()
                return

    
# --- Запуск ---
if __name__ == "__main__":
    app = AUVController(Config)
    while True:
        app.task1()
