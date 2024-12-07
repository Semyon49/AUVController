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

    LOWER_BOUND_ORANGE = np.array([10, 50, 50])
    UPPER_BOUND_ORANGE = np.array([35, 255, 255])
    
    LOWER_BOUND_RED1 = np.array([0, 120, 70])
    UPPER_BOUND_RED1 = np.array([10, 255, 255])

    LOWER_BOUND_RED2 = np.array([170, 120, 70])
    UPPER_BOUND_RED2 = np.array([180, 255, 255])


    AREA_THRESHOLD = 53000

    KP = 2.7
    KI = 0.1
    KD = 0.9

    DRAW = True

    # Settings folow line

    min_white = 80
    adaptiveThreshold = False
    limit_perc = 1.2
    last_perc = 10
    perc = 0

    
    # Driving settings
    turn_angle = 7
    shift_max = 15
    speed = 10
    rotation_coefficient = 2
    shift_coefficient = 10

    shift_step = 0.3
    turn_step = 10
    straight_run = 0.2
    find_turn_attempts = 5
    find_turn_step = 0.3
    max_steps = 100
