# -*- coding: utf-8 -*-
"""
Control functions
by Rida Lali, Pierre-Alexandre Peyronnet, Kevin Zhou
"""
from vars import *
from rplidar import RPLidar

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
