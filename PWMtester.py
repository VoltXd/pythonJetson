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

#boolean used to stop thread at the end
programEnded = False

def threadSerialPort(serialPort, tkWindow):
    """
    Enables to connect and disconnec serial ports while
    the program is running.

    Parameters
    ----------
    serialPort : serialwin32.Serial
        Serial port object used to communicate.
    tkWindow : Tk
        Current window.

    Returns
    -------
    None.

    """
    while not programEnded:
        #Check every ports connected
        comPortList = [p[0] for p in list(serial.tools.list_ports.comports())]
        
        #If nothing is connected, try to connect to ports until it's connected
        if (not serialPort.is_open):
            for port in comPortList:
                try:
                    serialPort.port = port
                    serialPort.open()
                    tkWindow.title("PWM tester - connected [{}]".format(port))
                except serial.SerialException:
                    print("Unable to open port " + port)
        #Else, if the connection has ended, close the port
        elif (serialPort.port not in comPortList):
            serialPort.close()
            tkWindow.title("PWM tester - connexion pending")
        #Wait to avoid spam
        time.sleep(0.1)
    
#Serial port declaration
sp = serial.Serial()
sp.baudrate = 115200

#Protocol declaration (name sucks + should not be a class now ?)
comm = connectionSPI.SpiProtocolAutonomousCar()


#Default values
pwmProp = 1500
pwmDir = 1150

#Limit values
propMax = 2000
propMin = 1000
dirMax = 1000
dirMin = 1300


def slideProp(var):
    """
    Function used when propulsion slider was moved.
    If a serial port is opened, send an order to change PWM.

    Parameters
    ----------
    var : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if sp.is_open:
        pwmProp = propulsion.get()
        pwmDir = direction.get()
        sp.write(comm.encodeMessage(pwmProp, pwmDir))
    
def slideDir(var):
    """
    Function used when direction slider was moved.
    If a serial port is opened, send an order to change PWM.

    Parameters
    ----------
    var : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if sp.is_open:
        pwmDir = direction.get()
        pwmProp = propulsion.get()
        sp.write(comm.encodeMessage(pwmProp, pwmDir))
    
def reset():
    """
    Function used when reset button was clicked.
    If a serial port is opened, send reset order.

    Returns
    -------
    None.

    """
    pwmProp = 1500
    pwmDir = 1150
    if sp.is_open:
        sp.write(comm.encodeMessage(pwmProp, pwmDir))
    propulsion.set(pwmProp)
    direction.set(pwmDir)

#Window size
width = 1500
height = 150

#Window Init.
root = Tk()
root.title("PWM tester - connexion pending")
root.geometry(str(width) + "x" + str(height))

#Thread Init.
threadSPManager = threading.Thread(target=threadSerialPort, args=(sp, root, ))
threadSPManager.start()

#Propulsion slider 
propulsion = Scale(root, from_=propMin, to=propMax, orient=HORIZONTAL, resolution=1, length=width*0.9, label='Propulsion', command=slideProp)
propulsion.pack()
propulsion.set(pwmProp)

#Direction slider
direction = Scale(root, from_=dirMin, to=dirMax, orient=HORIZONTAL, resolution=1, length=width*0.9, label='Direction', command=slideDir)
direction.pack()
direction.set(pwmDir)

#Reset button
resetButton = Button(root, command=reset, text='Reset')
resetButton.pack()
    
#infinite loop
root.mainloop()

#Closing program
if sp.isOpen():
    sp.close()          #Closing serial port
programEnded = True     #Closing thread
threadSPManager.join()