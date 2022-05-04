# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:38:26 2022

@author: Pierre-Alexandre
"""
#TODO: protocol ajouter KP et KI, terminal ASCII only
from tkinter import *
from tkinter import ttk
import threading
import time
import serial
import serial.tools.list_ports
import protocol

#boolean used to stop thread at the end
programEnded = False

#Serial port declaration
sp = serial.Serial()
sp.baudrate = 115200

#Protocol object declaration (class + file name suck)
comm = protocol.CarProtocol()

#Default values
pwmProp = 1500
pwmDir = 1150
speed0 = 0

#Limit values
propMax = 2000
propMin = 1000
dirMax = 1000
dirMin = 1300
speedMax = 8
speedMin = 0

#Variable for Terminal
terminalBufferSize = 128

#Functions
def threadUpdateTerminal(serialPort, tkTerminal, stringBuffer):
    while not programEnded:
        comPortList = [p[0] for p in list(serial.tools.list_ports.comports())]
        if serialPort.is_open and serialPort.port in comPortList:
            stringBuffer += serialPort.read_all().decode("ASCII")
            if len(stringBuffer) > terminalBufferSize:
                stringBuffer = stringBuffer[(len(stringBuffer) - terminalBufferSize):]
            if bool(tkTerminal.winfo_exists()):
                tkTerminal.delete(1.0, END)
                tkTerminal.insert(END, stringBuffer)
        time.sleep(0.1)
    

def threadSerialPort(serialPort, tkWindow, portCB):
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
    #The program crashes when it tries to acces destroyed widgets,
    #save previous ports to limit their acces.
    previousPorts = []
    
    #Infinite loop
    while not programEnded:
        #Check every ports connected, save them if they changed
        comPortList = [p[0] for p in list(serial.tools.list_ports.comports())]
        if not comPortList == previousPorts:
            #Change detected, modify combobox
            if len(comPortList) == 0:
                portCB['values'] = ["No port"]
                portCB.set("No port")
            else:
                portCB['values'] = comPortList
        previousPorts = comPortList.copy()
        
        #If nothing is connected, try to connect to ports until it's connected
        if (not serialPort.is_open):    
            for port in comPortList:
                try:
                    serialPort.port = port
                    serialPort.open()
                    tkWindow.title("PWM tester - connected [{}]".format(port))
                    portCB.set(port)
                except serial.SerialException:
                    print("Unable to open port " + port)
        #Else, if the selected port is not the connected one, change it
        elif portCB.get() not in [serialPort.port, "No port"]:
            port = portCB.get()
            try:
                serialPort.close()
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
        payload = (pwmProp, pwmDir)
        sp.write(comm.encodeMessage(payload))
    
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
        payload = (pwmProp, pwmDir)
        sp.write(comm.encodeMessage(payload))
    
def reset():
    """
    Function used when reset button was clicked.
    If a serial port is opened, send reset order.

    Returns
    -------
    None.

    """
    if comm.protocol == "PWM":
        pwmProp = 1500
    elif comm.protocol == "ASSERVISSEMENT":
        pwmProp = 0
    pwmDir = 1150
    if sp.is_open:
        payload = (pwmProp, pwmDir)
        sp.write(comm.encodeMessage(payload))
    propulsion.set(pwmProp)
    direction.set(pwmDir)
    
def onProtocolChange(evt):
    """
    

    Parameters
    ----------
    evt : Tkinter.event
        Tells many things.

    Returns
    -------
    None.

    """
    protocol = protocolCB.get()
    comm.setProtocol(protocol)
    if protocol == "PWM":
        propulsion.configure(to=propMax, from_=propMin, resolution=1, label='Propulsion')
        propulsion.set(pwmProp)
    elif protocol == "ASSERVISSEMENT":
        propulsion.configure(to=speedMax, from_=speedMin, resolution=0.001, label='Consigne de vitesse')
        propulsion.set(speed0)
    propulsion.pack()

    

#Window size
width = 700
height = 350

#Window Init.
root = Tk()
root.title("PWM tester - connexion pending")
root.geometry(str(width) + "x" + str(height))
root.resizable(width=False, height=False)

#Frames
lfCommand = LabelFrame(root, text="Commandes", padx=10, pady=10)
lfCommand.pack(side=BOTTOM, fill="both", expand="yes")

lfSettings = LabelFrame(root, text="Paramètres", padx=10, pady=10)
lfSettings.pack(anchor='nw', side=LEFT, fill="both", expand="yes")

lfTerminal = LabelFrame(root, text="Terminal", padx=10, pady=10)
lfTerminal.pack(anchor='ne', side=RIGHT, fill="both", expand="yes")

fSliders = Frame(lfCommand, relief=GROOVE, borderwidth=2)
fSliders.pack(padx = 10, pady=10)

fPort = Frame(lfSettings, relief=GROOVE, borderwidth=0)
fPort.pack(side=LEFT, anchor="nw", padx = 10, pady=0)

fProtocol = Frame(lfSettings, relief=GROOVE, borderwidth=0)
fProtocol.pack(side=RIGHT, anchor="ne", padx = 10, pady=0)

#Propulsion slider 
propulsion = Scale(fSliders, from_=propMin, to=propMax, orient=HORIZONTAL, resolution=1, length=width*0.95, label='Propulsion', command=slideProp)
propulsion.pack()
propulsion.set(pwmProp)

#Direction slider
direction = Scale(fSliders, from_=dirMin, to=dirMax, orient=HORIZONTAL, resolution=1, length=width*0.95, label='Direction', command=slideDir)
direction.pack()
direction.set(pwmDir)

#Reset button
resetButton = Button(lfCommand, command=reset, text='Reset')
resetButton.pack()

#COM PORT label
comportL = Label(fPort, text="Port", padx = 10, pady=10)
comportL.pack(side=TOP, anchor='nw')

#COM PORT combobox
comportCB = ttk.Combobox(fPort, state="readonly", values=["No port"])
comportCB.set("No port")
comportCB.pack(side=LEFT, anchor="nw")


#Protocol label
protocolL = Label(fProtocol, text="Protocole", padx = 10, pady=10)
protocolL.pack(anchor='nw')

#Protocol combobox
protocolCB = ttk.Combobox(fProtocol, state="readonly", values=["PWM", "ASSERVISSEMENT"])
protocolCB.set("PWM")
protocolCB.bind("<<ComboboxSelected>>", onProtocolChange)
protocolCB.pack(side=LEFT, anchor="nw")

#Terminal label
terminal = Text(lfTerminal, padx=10, pady=10, state='normal')
terminal.pack()

#Timer init.
terminalBuffer = ""
terminalUpdateThread = threading.Thread(target=threadUpdateTerminal, args=(sp, terminal, terminalBuffer, ))
terminalUpdateThread.start()

#Thread Init.
threadSPManager = threading.Thread(target=threadSerialPort, args=(sp, root, comportCB, ))
threadSPManager.start()

#infinite loop
root.mainloop()

#Closing program
if sp.isOpen():
    sp.close()          #Closing serial port
programEnded = True     #Closing thread
threadSPManager.join()
terminalUpdateThread.join()