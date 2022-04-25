# -*- coding: utf-8 -*-
"""
Control functions
by Rida Lali, Pierre-Alexandre Peyronnet, Kevin Zhou
"""
from vars import *
import spidev

# ____________ SPI DEFINITION ____________
BUS = 0
DEVICE = 0

SPI = spidev.SpiDev()
SPI.open(BUS, DEVICE)

SPI.max_speed_hz = 1000000
SPI.mode = 0

# SPI Function
def send_SPI(scaled_command):
    if scaled_command['neutral']:
        to_send = [115, 0 + 140]
    else:
        # Commands limits
        if scaled_command['reverse']:
            motor_speed = int(sat(scaled_command['throttle'], xmin=0, xmax=9))
        else:
            motor_speed = int(sat(scaled_command['throttle'], xmin=0, xmax=100))

        steering = int(sat(scaled_command['steering'], xmin=-20, xmax=20))

        # Sending commands
        if scaled_command['reverse']:
            to_send = [motor_speed, steering + 140] # TODO
        else:
            to_send = [motor_speed + 10, steering + 140] #TODO

            
    reply = SPI.xfer2(to_send)
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