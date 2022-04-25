# -*- coding: utf-8 -*-
"""
MultiThreading stuff
by Jamy Lafenetre, Rida Lali, Pierre-Alexandre Peyronnet
"""
from vars import *
from threading import Thread, Timer

from control import *

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