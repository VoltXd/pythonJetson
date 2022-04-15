# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:37:46 2022

@author: jamyl
"""
import numpy as np
from threading import Thread, Timer
import time
from queue import Queue
from stable_baselines3 import SAC
from rplidar import RPLidar
import sim_to_real_library as s2r
from pynput import keyboard
import spidev
from jamys_toolkit import lidar_formater, denormalize_action
from connectionSPI import SpiProtocolAutonomousCar


# _________ global variables ________________
TIME = time.time()
DECISION_PERIOD = 0.1  # seconds
OBSERVATION_PERIOD = 0.1
LAST_RUNNING_TIME = None # To check if the car is stuck
CRASH_TIMER = 1  # seconds
MODEL = SAC.load("/home/pi/Documents/vroom/Lidar_only")

THROTTLE_SCALE_FORWARD = 100
THROTTLE_SCALE_REVERSE = 9

STEERING_SCALE = 30
RECEIVED_MOTOR_SPEED = None

#Command max value
motorSpeedMaxValue = 1
steeringMaxValue = 0.5

#Speed PWM values
pwmSpeedZero = 1500
pwmSpeedMax = 1000

#Steering PWM values 
#(!!!!!!!!!!!!!! A REGLER !!!!!!!!!!!!!!!!!!)
#(!!!!!!!!!!!!!! A REGLER !!!!!!!!!!!!!!!!!!)
#(!!!!!!!!!!!!!! A REGLER !!!!!!!!!!!!!!!!!!)
#(!!!!!!!!!!!!!! A REGLER !!!!!!!!!!!!!!!!!!)
#(!!!!!!!!!!!!!! A REGLER !!!!!!!!!!!!!!!!!!)
#(!!!!!!!!!!!!!! A REGLER !!!!!!!!!!!!!!!!!!)
#Si les valeurs ne sont pas exactement bonnes
pwmSteerZero = 1150
pwmSteerMax = 1000

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

# ________ LIDAR DEFINITION _________________
PORT_NAME = "/dev/ttyUSB0"

LIDAR = RPLidar(PORT_NAME)


def start_lidar():
    try:
        info = start_lidar()
        print(info, type(info))
    except:
        start_lidar()


health = LIDAR.get_health()
print(health)

ITERATOR = LIDAR.iter_scans()  # Getting scan (type = generator)
OBSERVATION = None

# ____________ SPI DEFINITION ____________
BUS = 0
DEVICE = 0

SPI = spidev.SpiDev()
SPI.open(BUS, DEVICE)

SPI.max_speed_hz = 1000000
SPI.mode = 0

comm = SpiProtocolAutonomousCar()

# __________ Control functions _________________

def convertCommandToPWM(steering, motor_speed):
    speedPWM = int((pwmSpeedMax - pwmSpeedZero) * motor_speed / motorSpeedMaxValue + pwmSpeedZero)
    steerPWM = int((pwmSteerMax - pwmSteerZero) * steering / steeringMaxValue + pwmSteerZero)
    return (speedPWM, steerPWM)

def is_car_stuck(received_motor_speed):
    #TODO
    #remake it able to see if it's stuck 
    return False
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

def send_SPI(scaled_command):
    if scaled_command['neutral']:
        motor_speed, steering = 0, 0
    else:
        # Commands limits
        if scaled_command['reverse']:
            motor_speed = -scaled_command['throttle']
        else:
            motor_speed = scaled_command['throttle']

        steering = scaled_command['steering']

    pwmValues = convertCommandToPWM(steering, motor_speed)
    message = comm.encodeMessage(pwmValues[0], pwmValues[1])
    reply = SPI.xfer2(message)
    speed = reply[0]
    back_obstacle = reply[1]
    print("--back obstacle :", back_obstacle)
    global RECEIVED_MOTOR_SPEED, BACK_OBSTACLE
    RECEIVED_MOTOR_SPEED = speed # the reply is the measured motor speed
    if back_obstacle == 1:
        BACK_OBSTACLE = True
    else:
        BACK_OBSTACLE = False
    BACK_OBSTACLE = False
    
def push_action(command):
    send_SPI(command)

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


# ____________ MultiThreading stuff ____________________________

def decision_making_thread(obs_q):
    """ Choose an action based on observations

    Parameters
    ----------
    obs_q : Queue
        Queue containing the latest observation
    action_q : Queue
        Queue where the action will be passed

    Returns
    -------
    None.

    """
    # Calling back the same thread for periodic decision making

    Timer(DECISION_PERIOD, decision_making_thread, args=(obs_q,),).start()

    # getting the latest observation
    obs = obs_q.get()

    # deciding
    # print("Deciding", obs)
    command = decision_making(obs)

    # pushing action
    push_action(command)


def fetch_observation_thread(obs_q):
    """ Fetch the observation to feed the observation queue

    Parameters
    ----------
    obs_q : Queue
        queue containing the observation

    Returns
    -------
    None.
    """
    Timer(OBSERVATION_PERIOD, fetch_observation_thread, args=(obs_q,),).start()
    obs = fetch_observation()
    if obs_q.empty():  # Queue are FIFO. We are only using 1 element
        obs_q.put(obs)


def on_press(key):
    global STATE
    try:
        if key.char == "p":  # pause"
            STATE = STOP
            print("PAUSE")

        elif key.char == "s":  # start
            global IS_STARTING, IS_STARTING_INIT_TIME
            IS_STARTING = True
            IS_STARTING_INIT_TIME = time.time()
            STATE = DRIVING
            print("STARTING...")

    except AttributeError:
        print("special key {0} pressed".format(key))


# ______________ Thread initialisation and general stuffs _____________________
def main():
    # ______ Keyboard init ______________________
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # __________ Threads ___________________
    obs_q = Queue()

    decision_t = Thread(target=decision_making_thread, args=(obs_q,))
    obs_t = Thread(target=fetch_observation_thread, args=(obs_q,))

    decision_t.start()
    obs_t.start()


# _________ Starting main ______________________

main()