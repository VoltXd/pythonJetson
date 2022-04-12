# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:32:47 2021

@author: user
"""

import serial
import struct

class SpiProtocolAutonomousCar():
    def __init__(self, port, baudrate=9600):
        #self.sp = serial.Serial(port ,baudrate)
        return
    
    def __str__(self):
        return "Communication du coach\nPort: {}, BaudRate: {}".format(self.sp.port, self.sp.baudrate)
    
    #***************Frame Encoding***************
    #pwmPropulsion (int 2 Bytes) : Pwm propulsion value 
    #pwmDirection (int 2 Bytes) : Pwm direction value 
    def encodeMessage(self, pwmPropulsion, pwmDirection):
        msg = b'\xFF'
        msg += pwmPropulsion.to_bytes(2, 'big')
        msg += pwmDirection.to_bytes(2, 'big')
        msg += self.calculateChecksum(msg).to_bytes(1, 'big')
        
        #Envoie de la trame
        #self.sp.write(msg)
        #for b in msg:
        #    print(b)
        return msg
        
    def calculateChecksum(self, msg):
        checksum = 0
        for b in msg:
            checksum ^= b
        return checksum
        
    def closeConnection(self):
        #self.sp.close()
        return
    
if __name__ == "__main__":
    comm = SpiProtocolAutonomousCar("COM10", 115200)
    #comm.encodeAndSendMessage(1500, 1150)
    #comm.closeConnection()
    
    