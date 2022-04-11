# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:32:47 2021

@author: user
"""

import serial
import struct

class CoachCommunication():
    def __init__(self, port, baudrate=9600):
        #self.sp = serial.Serial(port ,baudrate)
        return
    
    def __str__(self):
        return "Communication du coach\nPort: {}, BaudRate: {}".format(self.sp.port, self.sp.baudrate)
    
    #***************Encodage des trames***************
    #robotID: Numéro du robot qui doit lire le message (0 ou 1)
    #charge: Commande de chargement du tir (0 ou 1)
    #kick: Commande de tir (0 ou 1)
    #dribble: Commande du dribbleur (Je sais pas trop ce qu'on mettra dedans mais pour l'instant c'est un entier entre 0 et 31)
    #vTan, vNorm, vAng: Vitesses (float)
    def encodeAndSendMessage(self, pwmPropulsion, pwmDirection):
        msg = b'\xFF'
        msg += pwmPropulsion.to_bytes(2, 'big')
        msg += pwmDirection.to_bytes(2, 'big')
        msg += self.calculateChecksum(msg).to_bytes(1, 'big')
        
        #Envoie de la trame
        #self.sp.write(msg)
        for b in msg:
            print(b)
        return msg
        
    def calculateChecksum(self, msg):
        checksum = 0
        for b in msg:
            checksum ^= b
        return checksum
        
    def closeCoach(self):
        self.sp.close()
        return
    
if __name__ == "__main__":
    comm = CoachCommunication("COM10", 115200)
    comm.encodeAndSendMessage(1500, 1150)
    #comm.closeCoach()
    
    