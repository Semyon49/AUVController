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

    LOWER_BOUND_RED = np.array([0, 10, 70])   
    UPPER_BOUND_RED = np.array([25, 256, 256])   


    AREA_THRESHOLD = 53000

    KP = 2.7
    KI = 0.1
    KD = 0.9

    DRAW = True