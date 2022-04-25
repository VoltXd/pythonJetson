# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:38:26 2022

@author: Pierre-Alexandre
"""

from tkinter import *
import threading
import time
import serial
import serial.tools.list_ports
import connectionSPI

programEnded = False

def threadSerialPort(serialPort, tkWindow):
    while not programEnded:
        comPortList = [p[0] for p in list(serial.tools.list_ports.comports())]
        if (not serialPort.is_open):
            for port in comPortList:
                try:
                    serialPort.port = port
                    serialPort.open()
                    tkWindow.title("PWM tester - connected [{}]".format(port))
                except serial.SerialException:
                    print("Unable to open port " + port)
        elif (serialPort.port not in comPortList):
            serialPort.close()
            tkWindow.title("PWM tester - connexion pending")
        time.sleep(0.1)
    
sp = serial.Serial()
sp.baudrate = 115200

comm = connectionSPI.SpiProtocolAutonomousCar()


pwmProp = 1500
pwmDir = 1150

propMax = 2000
propMin = 1000
dirMax = 1000
dirMin = 1300


def slideProp(var):
    pwmProp = propulsion.get()
    pwmDir = direction.get()
    if sp.is_open:
        sp.write(comm.encodeMessage(pwmProp, pwmDir))
    
def slideDir(var):
    pwmDir = direction.get()
    pwmProp = propulsion.get()
    if sp.is_open:
        sp.write(comm.encodeMessage(pwmProp, pwmDir))
    
def reset():
    pwmProp = 1500
    pwmDir = 1150
    if sp.is_open:
        sp.write(comm.encodeMessage(pwmProp, pwmDir))
    propulsion.set(pwmProp)
    direction.set(pwmDir)

width = 1500
height = 150

root = Tk()
root.title("PWM tester - connexion pending")
root.geometry(str(width) + "x" + str(height))

threadSPManager = threading.Thread(target=threadSerialPort, args=(sp, root, ))
threadSPManager.start()

propulsion = Scale(root, from_=propMin, to=propMax, orient=HORIZONTAL, resolution=1, length=width*0.9, label='Propulsion', command=slideProp)
propulsion.pack()
propulsion.set(pwmProp)

direction = Scale(root, from_=dirMin, to=dirMax, orient=HORIZONTAL, resolution=1, length=width*0.9, label='Direction', command=slideDir)
direction.pack()
direction.set(pwmDir)

myButton = Button(root, command=reset, text='Reset').pack()
    
    

root.mainloop()
if sp.isOpen():
    sp.close()
programEnded = True    
threadSPManager.join()