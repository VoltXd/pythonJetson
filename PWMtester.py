# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:38:26 2022

@author: Pierre-Alexandre
"""

from tkinter import *
import PIL 
import serial
import connectionSPI

sp = serial.Serial('COM4', 115200)
comm = connectionSPI.SpiProtocolAutonomousCar()

pwmProp = 1500
pwmDir = 1150

propMax = 2000
propMin = 1000
dirMax = 1000
dirMin = 1300


def slideProp(var):
    pwmProp = propulsion.get()
    sp.write(comm.encodeMessage(pwmProp, pwmDir))
    
def slideDir(var):
    pwmDir = direction.get()
    sp.write(comm.encodeMessage(pwmProp, pwmDir))
    
def reset():
    pwmProp = 1500
    pwmDir = 1150
    sp.write(comm.encodeMessage(pwmProp, pwmDir))
    propulsion.set(pwmProp)
    direction.set(pwmDir)

width = 1500
height = 300

root = Tk()
root.title("PWM tester")
root.iconbitmap("")
root.geometry(str(width) + "x" + str(height))

propulsion = Scale(root, from_=propMin, to=propMax, orient=HORIZONTAL, resolution=1, length=width*0.9, label='Propulsion', command=slideProp)
propulsion.pack()
propulsion.set(pwmProp)

direction = Scale(root, from_=dirMin, to=dirMax, orient=HORIZONTAL, resolution=1, length=width*0.9, label='Direction', command=slideDir)
direction.pack()
direction.set(pwmDir)

myButton = Button(root, command=reset, text='Reset').pack()
    
    

root.mainloop()
sp.close()