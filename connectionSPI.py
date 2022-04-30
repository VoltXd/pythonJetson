# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:32:47 2021

@author: user
"""

import struct

class SpiProtocolAutonomousCar():
    """Class used for communication protocol"""
    PROTOCOLS = ("PWM", "ASSERVISSEMENT")
    
    def __init__(self, protocol='PWM'):
        """
        

        Parameters
        ----------
        protocol : str, optional
            Protocol used. The default is 'PWM'.

        Returns
        -------
        None.

        """
        if protocol in self.PROTOCOLS:
            self.protocol = protocol
        else:
            self.protocol = "PWM"
        return
    
    #***************Frame Encoding***************
    #pwmPropulsion (int 2 Bytes) : Pwm propulsion value 
    #pwmDirection (int 2 Bytes) : Pwm direction value 
    def encodeMessage(self, payload):
        """
        Encode a message according to the protocol selected

        Parameters
        ----------
        payload : tuple
            Contains the variables to transmit.

        Returns
        -------
        msg : bytes
            The message to transmit.

        """
        if self.protocol == "PWM":
            pwmPropulsion = payload[0]
            pwmDirection = payload[1]
            msg = b'\xFF'
            msg += pwmPropulsion.to_bytes(2, 'big')
            msg += pwmDirection.to_bytes(2, 'big')
            msg += self.calculateChecksum(msg).to_bytes(1, 'big')
        elif self.protocol == "ASSERVISSEMENT":
            speedCommand = payload[0]
            pwmDirection = payload[1]
            msg = b'\xFF'
            msg += struct.pack('f', speedCommand)
            msg += pwmDirection.to_bytes(2, 'big')
            msg += self.calculateChecksum(msg).to_bytes(1, 'big')            
        
        #Envoie de la trame
        #self.sp.write(msg)
        #for b in msg:
        #    print(b)
        return msg
    
    def setProtocol(self, protocol):
        """
        Sets the protocol

        Parameters
        ----------
        protocol : str
            Name of the protocol to use.

        Returns
        -------
        None.

        """
        if protocol in self.PROTOCOLS:
            self.protocol = protocol
        return
        
    
    def calculateChecksum(self, msg):
        """
        Calculate the checksum of the frame, using 8 bits XOR

        Parameters
        ----------
        msg : bytes
            Message without checksum.

        Returns
        -------
        checksum : bytes
            Result of the 8XOR.

        """
        checksum = 0
        for b in msg:
            checksum ^= b
        return checksum
    
if __name__ == "__main__":
    comm = SpiProtocolAutonomousCar()
    #comm.encodeAndSendMessage(1500, 1150)
    #comm.closeConnection()
    
    