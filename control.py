# -*- coding: utf-8 -*-
"""
Control functions
by Jamy Lafenetre, Rida Lali, Kevin Zhou
"""
from vars import *
import numpy as np

import sim_to_real_library as s2r
from sac_toolkit_client import lidar_formater, denormalize_action

from spi_tools import *
from lidar_tools import *
import time

# __________ Control functions _________________
def is_car_stuck(received_motor_speed):
    if STATE == STOP:
        return False
    global LAST_RUNNING_TIME
    if RECEIVED_MOTOR_SPEED is None:
        print("I have no speed measure bro WTF ??")
        return False
    if received_motor_speed > 0 or LAST_RUNNING_TIME is None: #  running or init
        LAST_RUNNING_TIME = time.time()
        return False
    if time.time() - LAST_RUNNING_TIME >= CRASH_TIMER:
        LAST_RUNNING_TIME = None
        return True
    return False

def sat(x, xmin, xmax):
    """ saturation function

    Parameters
    ----------
    x : float
        Input number
    xmin : float
        low saturation bound
    xmax : float
        high saturation bound

    Returns
    -------
    float
        Satured version of x

    """
    if x > xmax:
        return xmax
    if x < xmin:
        return xmin
    return x

def init_evasive_maneuver():
    global EVASIVE_MANEUVER_INIT_TIME
    if EVASIVE_MANEUVER_INIT_TIME is None: # first time in the function
        EVASIVE_MANEUVER_INIT_TIME = time.time()
        
    if time.time() - EVASIVE_MANEUVER_INIT_TIME > NEUTRAL_DELAY:
        # exiting init state
        global STATE
        STATE = EVASIVE_MANEUVER
        EVASIVE_MANEUVER_INIT_TIME = None
    
    command = {'neutral':True}
    return command


def evasive_maneuver(obs):
    global EVASIVE_MANEUVER_START_TIME
    if EVASIVE_MANEUVER_START_TIME is None: # first time in the function
        EVASIVE_MANEUVER_START_TIME = time.time()
    
    if (time.time()-EVASIVE_MANEUVER_START_TIME > EVASIVE_MANEUVER_DURATION) or BACK_OBSTACLE:
        global STATE, MEAN_STEERING
        STATE = DRIVING
        MEAN_STEERING = 0
        EVASIVE_MANEUVER_START_TIME = None
    
    action = denormalize_action(MODEL.predict(obs, deterministic=True)[0])
    command = {'throttle':1,
               'steering':-action[1],
               'reverse' : True,
               'neutral':False}
    
#     command = {'throttle':1,
#                'steering':0,
#                'reverse':True,
#                'neutral':False} # TODO
    
    return command

def decision_making(obs):
    print("\nstate : ", STATE)
    if STATE == DRIVING:
        action = denormalize_action(MODEL.predict(obs, deterministic=True)[0])
        command={'throttle':action[0], 'steering':action[1], 'reverse':False, 'neutral':False}
        global MEAN_STEERING
        MEAN_STEERING = STEERING_MEAN_COEF * command['steering'] + (1- STEERING_MEAN_COEF)*MEAN_STEERING
        command['steering'] = MEAN_STEERING

    elif STATE == STOP:
        command = {'throttle':0, 'steering':0, 'reverse':False, 'neutral':False}  # 0 speed, 0 steering

    elif STATE == EVASIVE_MANEUVER:
        print("evasive maneuver like a boss")
        command = evasive_maneuver(obs)
        
    elif STATE == INIT_EVASIVE_MANEUVER:
        print("init maneuver")
        command = init_evasive_maneuver()
        
    else:
        raise ValueError(
            """ STATE took unexpected value. Expected {}, {},{}
                         or {}, but received {}""".format(
                DRIVING, STOP, EVASIVE_MANEUVER, INIT_EVASIVE_MANEUVER, STATE
            )
        )
    return command

def scale_command(command):
    """
    mapping of:
        throttle forward: [0, 1] -> [0, THROTTLE_SCALE_FORWARD]
        throttle backward: [0, 1] -> [0, THROTTLE_SCALE_REVERSE]
        steering : [-0.5, 0.5] -> [-STEERING_SCALE, STEERING_SCALE]
    """

    if command['reverse']:
        scaled_command={'throttle':THROTTLE_SCALE_REVERSE * abs(command['throttle']),
                        'steering':STEERING_SCALE * command['steering'],
                        'reverse' :command['reverse'],
                        'neutral' :command['neutral']}
    else:
        scaled_command={'throttle':THROTTLE_SCALE_FORWARD * abs(command['throttle']),
                'steering':STEERING_SCALE * command['steering'],
                'reverse' :command['reverse'],
                'neutral' :command['neutral']}
    
    return scaled_command


def push_action(command):
    if command['neutral']:
        send_SPI(command)
    else:
        scaled_command = scale_command(command)
        send_SPI(scaled_command)


def fetch_observation():
    scan = next(ITERATOR)
    dots = np.array([(meas[1], meas[2]) for meas in scan])  # Convert data into np.array

    current_lidar, conversion_error = lidar_formater(s2r.lidar_real_to_sim(dots), 100)
    if conversion_error:
        raise ValueError("No lidar data was received")

    global OBSERVATION
    if OBSERVATION is None:  # init
        OBSERVATION = {
            "current_lidar": current_lidar,
            "prev_lidar1": np.copy(current_lidar),
            "prev_lidar2": np.copy(current_lidar),
            "prev_lidar3": np.copy(current_lidar),
            "prev_lidar4": np.copy(current_lidar),
            "prev_lidar5": np.copy(current_lidar),
            "prev_throttle": np.array([0]),
            "prev_steering": np.array([0]),
        }
    else:
        prev_lidar5 = OBSERVATION["prev_lidar4"]
        prev_lidar4 = OBSERVATION["prev_lidar3"]
        prev_lidar3 = OBSERVATION["prev_lidar2"]
        prev_lidar2 = OBSERVATION["prev_lidar1"]
        prev_lidar1 = OBSERVATION["current_lidar"]

        OBSERVATION = {
            "current_lidar": current_lidar,
            "prev_lidar1": prev_lidar1,
            "prev_lidar2": prev_lidar2,
            "prev_lidar3": prev_lidar3,
            "prev_lidar4": prev_lidar4,
            "prev_lidar5": prev_lidar5,
            "prev_throttle": np.array([0]),
            "prev_steering": np.array([0]),
        }

    global STATE, IS_STARTING
    if IS_STARTING_INIT_TIME is None or time.time() - IS_STARTING_INIT_TIME >IS_STARTING_DURATION:
        IS_STARTING = False
        
    if not IS_STARTING:
        stuck_condition = is_car_stuck(RECEIVED_MOTOR_SPEED)
        
    else:
        stuck_condition = False
        
    if stuck_condition and STATE != EVASIVE_MANEUVER and STATE != INIT_EVASIVE_MANEUVER:
        print("Help im stuck : state ",STATE)
        STATE = INIT_EVASIVE_MANEUVER


    return OBSERVATION