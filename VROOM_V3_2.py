import numpy as np
import time
import matplotlib.pyplot as plt
import random
import spidev
from pynput import keyboard
from rplidar import RPLidar
import connectionSPI

''' SPI DEFINITION '''
bus = 0
device = 0

spi = spidev.SpiDev()
spi.open(bus, device)

spi.max_speed_hz = 1000000
spi.mode = 0

''' LIDAR DEFINITION '''
PORT_NAME = '/dev/ttyUSB0'

lidar = RPLidar(PORT_NAME)

info = lidar.get_info()
print(info, type(info))

health = lidar.get_health()
print(health, type(health))

''' STATES '''
STATE = 0
STOP = 0
DRIVING = 1
EVASIVE_MANEUVER = 2



''' CONSTANTS '''
#steer_coeff :
K_p = 1/20
K_d = 1/100000000
speed_coeff = 1/120
angle_coeff = 0.07
detection_angle_min = 45
detection_angle_max = 90
crash_time_limit = 0

''' FUNCTIONS '''
def car_ctrl(steering, motor_speed):
    
    #Commands limits
    if motor_speed > 10:
        motor_speed = 10
        
    if steering > 15:
        steering =15
    elif steering < -15:
        steering = -15
    
    #Sending commands 
    to_send = [motor_speed, steering] 
    reply = spi.xfer2(to_send)
    speed_meas = reply[1] + (reply[0] << 8)
    return speed_meas

def find_detection_angle(front_dist):
    ''' find the right detection angle depending on the distance in front of the car '''
    detection_angle = -angle_coeff*(front_dist-600) + detection_angle_max
            
    if detection_angle < detection_angle_min:
        detection_angle = detection_angle_min
                
    elif detection_angle > detection_angle_max:
        detection_angle = detection_angle_max
    
    return detection_angle

        
'''KEYBOARD CONNECTION'''
def on_press(key):
    global STATE
    global STOP
    global DRIVING
    global EVASIVE_MANEUVER
    try:
        if key.char =='p': #pause"
            STATE = STOP
            print("PAUSE")
            V = car_ctrl(0, 0)
            
        elif key.char == 's': #start
            STATE = DRIVING
            print("STARTING...")
            
            
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


listener = keyboard.Listener(
on_press=on_press)
listener.start()
    



def vroom():
    try:
        print('Ctrl+C to stop')
        iterator = lidar.iter_scans() #Getting scan (type = generator)
    
        old_err = 0
        old_steering = 0
        old_time = None
        speed_meas = 0
        old_speed_meas = None
        
        
        global crash_time_limit
        global STATE
        global STOP
        global DRIVING
        global EVASIVE_MANEUVER
        while True:
            
            if STATE == STOP:
                scan = next(iterator)
                old_err = 0
                old_steering = 0
                old_time = None
                speed_meas = 0
                old_speed_meas = None
                
            elif STATE == DRIVING:
                now = time.time()
                
                if old_time is None:
                    old_time = now
                    continue
                dt = now - old_time
                
                
                scan = next(iterator) #Get data
                dots = np.array([(meas[1], meas[2]) for meas in scan]) #Convert data into np.array
                
                #Find front distance
                front_dists = np.concatenate((dots[:,1][dots[:,0] < 10],dots[:,1][dots[:,0] > 350]))
                front_dist = np.mean(front_dists)
                
                
                
                if old_speed_meas is None:
                    old_speed_meas = speed_meas
                    continue
                    
                if speed_meas == 0 and old_speed_meas != 0:
                    crash_time_limit = now + 10
                else :
                    crash_time_limit = now
                    
                #STATE CHANGE
                if now > crash_time_limit:
                    STATE = EVASIVE_MANEUVER
                    print("evasive maneuver")
                
                
                #Find detection angle 
                detection_angle = find_detection_angle(front_dist)
                
                #Find distance on the left
                left_dists = dots[:,:][detection_angle+5 > dots[:,0]]
                left_dists = left_dists[:,1][left_dists[:,1] > detection_angle-5]
                left_dist = np.mean(left_dists)
                
                #Find distance on the right
                right_dists = dots[:,:][dots[:,0] > 355-detection_angle ]
                right_dists = right_dists[:,1][right_dists[:,0] < 365-detection_angle ]
                right_dist = np.mean(right_dists)
                
                
                if front_dists.size == 0:
                    motor_speed = 0
                else:
                    motor_speed = min(speed_coeff*front_dist,8)
                
                
                
                new_err = left_dist - right_dist
                if right_dists.size == 0:
                    steering = 15
                elif left_dists.size == 0:
                    steering = -15
                else:
                    #steering = 2*(K_p + K_d/dt*(new_err - old_err)) - old_steering #: TUSTIN
                    steering = K_p*new_err + K_d/dt*(new_err - old_err) #: EULER
                    old_err = new_err
                    print("in")
                
                
                print(int(steering), int(motor_speed), speed_meas)
                speed_meas = car_ctrl(int(steering), int(motor_speed))
                old_time = now
                old_front_dist = front_dist
                old_steering = steering

                    
                
            
            elif STATE == EVASIVE_MANEUVER:
                v = 0
            '''
                scan = next(iterator) #Get data
                dots = np.array([(meas[1], meas[2]) for meas in scan]) #Convert data into np.array
                #Find distance on all the left
                left_dists = dots[:,1][dots[:,0] < 90]
                left_dist = np.mean(left_dists)
                
                #Find the distance on all the right
                right_dists = dots[:,1][dots[:,0] > 270]
                right_dist = np.mean(right_dists)
                    
                if left_dist < right_dist:
                    steering = 15
                        
                else:
                    steering = -15
                
                motor_speed = 5
                
                t_end = time.time() + 1

                while time.time() < t_end:
                    car_ctrl(steering, motor_speed)
                
                #STATE CHANGE
                STATE = DRIVING
            '''
            
    except KeyboardInterrupt:
        print('Stoping...')
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    V = car_ctrl(0, 0)
    

if __name__ == '__main__':
    #vroom()        
    protocol = SpiProtocolAutonomousCar("bite")
    spi.xfer2(protocol.encodeMessage(1400, 1000))


