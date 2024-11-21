import cv2
import time
import numpy as np

from settings import Config
from processing import ImageProcessor, ImageProcessorInterface
from controllers import MotionController, DepthController

class AUVController(MotionController, DepthController, ImageProcessor):
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
            current_depth = self.config.AUV.get_depth()
            power = self.control_depth(current_depth)
            self.swim_forward(power)
            print(power)
            if abs(3.4 - current_depth) < 0.2:
                self.set_power_swim_up(-power) 
                self.stop()
                break

        while True:
            image = self.config.AUV.get_image_bottom()
            pt1, pt2 = self.detection_figures(image, self.config.LOWER_BOUND_ORANGE, self.config.UPPER_BOUND_ORANGE)
            area = ImageProcessorInterface.calculate_area(pt1, pt2)
            
            self.set_power_swim_forward(self.config.SPEED)
            if area > self.config.AREA_THRESHOLD:
                self.stop()
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
        while True:
            self.set_power_swim_forward(self.config.SPEED)
            image = self.config.AUV.get_image_bottom()
            x, y = self.detection_figures_bin(image, self.config.LOWER_BOUND_RED, self.config.UPPER_BOUND_RED)

            if x == 0 and y == 0:
                continue
            
            if abs(self.config.IMAGE_SHAPE[0] - y) < 7 and abs(self.config.IMAGE_SHAPE[1] - x) < 7:
                break 

            self.adjust_power_to_center((x, y))
            

            self.set_target_depth(0)
        while True:
            current_depth = self.config.AUV.get_depth()
            power = self.control_depth(current_depth)
            self.set_power_swim_up(power)

            if current_depth == 0:
                self.stop()
                return

    
# --- Запуск ---
if __name__ == "__main__":
    app = AUVController(Config)
    app.set_target_depth(3.4)
    app.corpus_research()
    # app.ascent()