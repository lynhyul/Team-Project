#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from pyfirmata import Arduino, util
import rospy
import time
from time import sleep
from std_msgs.msg import String, Float32

print("init")
#arduino
board = Arduino('/dev/ttyACM0')
sleep(0.5)

#controll values
FORWARD = 1
BACKWARD = 0
LEFT = -2
RIGHT = 2
MotorSpeed = 150
Hub = 77
steering_p = 0
r_mid_pos = 80
#Mode
Controll = 0
Linetrace = 1
Auto = 2
Mode = Controll


#Controll func
def go_Motor(Dir,Speed):
    pin_in1.write(Dir)
    pin_in2.write(abs(Dir-1))
    Speed = float(Speed)/255
    pin_ina.write(Speed)

def go_Motor_power(Dir,Speed,Steer_ang):
    global Hub
    pin_in1.write(Dir)
    pin_in2.write(abs(Dir-1))
    power = abs((Steer_ang - Hub))
    Speed = float((Speed+power))/255
    pin_ina.write(Speed)

def steer(Dir):
    global Hub
    if Dir == LEFT:
        pin_s.write(Hub - 65)
    if Dir == RIGHT:
        pin_s.write(Hub + 30)
    if Dir == FORWARD:
        pin_s.write(Hub)

def steer_direct(steering):
    pin_s.write(steering)

#Pins
pin_s = board.get_pin('d:9:s')
pin_in1 = board.get_pin('d:2:o')
pin_in2 = board.get_pin('d:7:o')
pin_ina = board.get_pin('d:6:p')

it = util.Iterator(board)
it.start()

#Motor init
go_Motor(FORWARD,0)
pin_s.write(Hub)

def drive_controll(res):
    global FORWARD
    global BACKWARD
    global LEFT
    global RIGHT
    global MotorSpeed
    #controll
    if res == 'S':
        go_Motor(FORWARD, 0)
        steer(FORWARD)
    if res == 'F':
        go_Motor(FORWARD, MotorSpeed)
        steer(FORWARD)
    if res == 'B':
        go_Motor(BACKWARD, MotorSpeed)
        steer(FORWARD)
    if res == 'L':
        steer(LEFT)
    if res == 'R':
        steer(RIGHT)
    if res == 'G':
        steer(LEFT)
        go_Motor(FORWARD, MotorSpeed)
    if res == 'I':
        steer(RIGHT)
        go_Motor(FORWARD, MotorSpeed)
    if res == 'H':
        steer(LEFT)
        go_Motor(BACKWARD, MotorSpeed)
    if res == 'J':
        steer(RIGHT)
        go_Motor(BACKWARD, MotorSpeed)
    #speed
    if res == '0':
        MotorSpeed = 150
    if res == '1':
        MotorSpeed = 160
    if res == '2':
        MotorSpeed = 170
    if res == '3':
        MotorSpeed = 180
    if res == '4':
        MotorSpeed = 190
    if res == '5':
        MotorSpeed = 200
    if res == '6':
        MotorSpeed = 210
    if res == '7':
        MotorSpeed = 220
    if res == '8':
        MotorSpeed = 230
    if res == '9':
        MotorSpeed = 240
    if res == 'q':
        MotorSpeed = 250

def set_driving_params(res):
    global FORWARD
    global BACKWARD
    global LEFT
    global RIGHT
    global MotorSpeed
    global r_mid_pos
    #controll
    if res == 'S':
        go_Motor(FORWARD, 0)
        steer(FORWARD)
    if res == 'L':
        r_mid_pos = 240
    if res == 'R':
        r_mid_pos = 80
    if res == 'B':
        r_mid_pos = 150
    #speed
    if res == '0':
        MotorSpeed = 150
    if res == '1':
        MotorSpeed = 160
    if res == '2':
        MotorSpeed = 170
    if res == '3':
        MotorSpeed = 180
    if res == '4':
        MotorSpeed = 190
    if res == '5':
        MotorSpeed = 200
    if res == '6':
        MotorSpeed = 210
    if res == '7':
        MotorSpeed = 220
    if res == '8':
        MotorSpeed = 230
    if res == '9':
        MotorSpeed = 240
    if res == 'q':
        MotorSpeed = 250

def drive_linetrace(mid_line):
    global FORWARD
    global MotorSpeed
    global Hub
    global steering_p
    # Instant
    k_p = 0.01
    global r_mid_pos
    # Variables
    steering = 0
    steering_ang = 0
    
    # Calculate steering
    steering = k_p*(mid_line - r_mid_pos)
    steering_p = steering

    if steering < -1:
        steering_ang = Hub - 30*1.5

    if steering >= -1 and steering < 0:
        steering_ang = steering*30*1.5 + Hub
    if steering == 0:
        steering_ang = steering*30 + Hub
    if steering <= 1 and steering > 0:
        steering_ang = steering*30 + Hub

    if steering > 1:
        steering_ang = Hub + 30

    print(steering_ang)
    steer_direct(steering_ang)
    #sleep(0.04)
    go_Motor(FORWARD, MotorSpeed)
    #sleep(0.13)
    #go_Motor(FORWARD, 0)

    



t0 = time.time()

#Subscriber
def callback(data):
    global t0
    global Mode
    global Controll

    #Mode Selection
    if data.data == 'W' or data.data == 'w':
        Mode = Controll
    elif data.data == 'U' or data.data == 'u':
        Mode = Linetrace
    elif data.data == 'V' or data.data == 'v':
        Mode = Auto
    #drive
    if Mode == Controll:
        drive_controll(data.data)
    if Mode == Linetrace:
        set_driving_params(data.data)

    t1 = time.time()
    #print("input:{0} period:{1}".format( data.data, (t1-t0) ))
    t0 = t1

def callback2(data):
    global Mode
    global Linetrace
    global Auto
    
    print("steering input:{0}".format(data.data))
    if Mode == Linetrace:
        drive_linetrace(data.data)
    if Mode == Auto:
        pass
    
def listener():
    rospy.init_node('listener', anonymous = True)

    rospy.Subscriber("chatter", String, callback)
    rospy.Subscriber("lane_info", Float32, callback2)

    rospy.spin() #listening loop

#main
if __name__=='__main__':
    listener()

