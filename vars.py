# -*- coding: utf-8 -*-
"""
Global variables file
by Jamy Lafenetre, Rida Lali, Pierre-Alexandre Peyronnet
"""
import time
from stable_baselines3 import SAC


# _________ global variables ________________
TIME = time.time()
DECISION_PERIOD = 0.1  # seconds
OBSERVATION_PERIOD = 0.1
LAST_RUNNING_TIME = None # To check if the car is stuck
CRASH_TIMER = 1  # seconds
FILE_MODEL = "/home/jetson/Documents/vroom/Lidar_only"
MODEL = SAC.load(FILE_MODEL)

THROTTLE_SCALE_FORWARD = 100
THROTTLE_SCALE_REVERSE = 9

STEERING_SCALE = 30
RECEIVED_MOTOR_SPEED = None

# ________ States ______________
STOP = 0
DRIVING = 1
EVASIVE_MANEUVER = 2
INIT_EVASIVE_MANEUVER = 3

STATE = STOP

EVASIVE_MANEUVER_INIT_TIME = None # To control the init duration
NEUTRAL_DELAY = 0.5 # in seconds. How long is the transition ?

EVASIVE_MANEUVER_DURATION = 1 # in seconds. How long is the maneuver ?                                                                                                               
EVASIVE_MANEUVER_START_TIME = None # to control the maneuver time
BACK_OBSTACLE = False # whether or not an obstacle was detected in the back

IS_STARTING = False # When true, is_car_crashed() will be ignored
IS_STARTING_DURATION = 0.5
IS_STARTING_INIT_TIME = None

# _______________

MEAN_STEERING = 0
STEERING_MEAN_COEF = 1 # must be in ]0, 1]