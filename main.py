# -*- coding: utf-8 -*-
"""
Main file for execution
by Jamy Lafenetre, Rida Lali, Pierre-Alexandre Peyronnet
"""
from threading import Thread, Timer
import time
from queue import Queue
from pynput import keyboard
from vars import *
from multithread import *

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