#!/usr/bin/env python 

import time
from time import sleep
import serial
import rospy
from std_msgs.msg import String

#serial
ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    )
#timer
t0 = time.time()

def printime(t1):
    global t0
    print("spend:{0}".format( (t1-t0) ))

def talker():
    global t0
    pub = rospy.Publisher('chatter',String,queue_size=3)
    rospy.init_node('talker', anonymous = True)
    rate = rospy.Rate(60) #60hz
    while True:
        if ser.readable():
            t1 = time.time()
            res = ser.read().decode()
            printime(t1)
            if not rospy.is_shutdown():
                if (res == 'X') or (res == 'x'):
                    pub.publish(res)
                    #rate.sleep()
                    print("send shutdown")
                elif (t1-t0) > 0.19: 
                    print("send key")
                    pub.publish(res)
                    t0 = t1


if __name__=='__main__':
      try:
          talker()
      except rospy.ROSInterruptException:
            pass
