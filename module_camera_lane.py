#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import threading
import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Int32, Float32

import glob
from find_lane_run import LaneFinder, DashboardCamera
import matplotlib.pyplot as plt
import numpy as np
import symfit
from imageio.core import NeedDownloadError
from typing import List
from windows_chung import Window, filter_window_list, joint_sliding_window_update, window_image, sliding_window_update

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/usr/local/lib')
import cv2

def shutdown():
    cap.release()
    #cv2.destroyWindow("image")
    #print("종료완료")
    rospy.sleep(1)

# laod info data
file_r = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/textfile.txt','r')

L = 0
'''while True:
    line = file_r.readline()
    try:escape = line.index('\n')
    except:escape=len(line)
    if line:
	L=int(line[0:-3])
    else:
        break'''
t_count = L
print("t_count: {0}".format(t_count))

# Finder param
global camera
global pub
  
# Set cameara node
camera_num = int(sys.argv[1])

print("Camera port: {0}".format(camera_num))

rospy.init_node('camera')    
rospy.loginfo('camera 노드 시작')
rospy.on_shutdown(shutdown)

img_pub = rospy.Publisher("image_raw", Image, queue_size=1)
#pub = rospy.Publisher('lane_counter',Int32,queue_size=1)
pub = rospy.Publisher('lane_info',Float32,queue_size=1)

img = None

bridge = CvBridge()

cap = cv2.VideoCapture(camera_num) # ls -lrt /dev/video* -> 0번 장치

rate = rospy.Rate(60)

#cap.set(cv2.CAP_PROP_FPS,30)
time.sleep(1)

# Thread camera Function
def cameara():
    global img

    while(True):
        while(True):
            ret, img = cap.read()
            if(ret==True):
                break
        print("is capturing")
        rate.sleep()

# Capture Function
def capture(res):
    global cap
    global t_count
    global img

    if (res == 'X') or (res == 'x'):
        shutdown()
        print('node shutdown')

    if(res == 'F'):
        if (t_count == 1000001):
            t_count = 0
        t_count += 1
        #img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
        #img_pub.publish(img_msg)
        #pub.publish(t_count)
        lane(t_count)
        print("Send position")
    #cv2.waitKey(10)

def lane(data):
    global camera
    global pub
    global img

    vector = time.time()
    #print('lane is run, data: {0}'.format(data))

    # laod last mid pos
    last_mid_pos = 0
    file_mid_log = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/mid_line_log/mid_log.txt','r')
    while True:
        line = file_mid_log.readline()
        if not line: break
        last_mid_pos = int(line)
    #print("last mid-line position: {0}".format(last_mid_pos))
    file_mid_log.close()

    #img = cv2.imread("/home/autocar/camera_im/camera_image"+str(324)+".jpeg")
    # image load
    test_img = cv2.resize(img, dsize=(320, 180), interpolation=cv2.INTER_AREA)
    test_img_copy = np.array(test_img).astype(np.uint8)
    

    # find lane
    lane_finder = LaneFinder(camera)  # need new instance per image to prevent smoothing
    mid_line = lane_finder.get_mid_line(test_img_copy, last_mid_pos)
    mid_line = round(mid_line,-1)
    

    # publish data
    pub.publish(mid_line)
    #print("send mid_line: {0}".format(mid_line))

   
    # log mid line
    
    file_a = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/textfile.txt','a') # history of mid
    file_a.write('%d %f\n'%(data, mid_line))

    file_mid_log = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/mid_line_log/mid_log.txt','w') # lastest mid log
    file_mid_log.write('%d'%(mid_line))

    print("process time:{0}".format(time.time()-vector))

    # save img
    '''
    # overhead view
    img_dash_undistorted = camera.undistort(test_img_copy)
    img_overhead = camera.warp_to_overhead(img_dash_undistorted)
    cv2.imwrite("/home/autocar/camera_im/camera_image"+str(data)+".jpeg",img_overhead)
    '''
    # normal view
    cv2.imwrite("/home/autocar/camera_im/camera_image"+str(data)+".jpeg",test_img_copy)

    #lane_finder.viz_pipeline(test_img_copy)
    #plt.show()



def lane_init():
    global camera

    # Calibrate using checkerboard
    calibration_img_files = glob.glob('/home/autocar/catkin_ws/src/msg_ss/scripts/data/camera_cal_low/*.jpg')
    #lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    lane_shape = [(55, 80), (265, 80), (0, 165), (320, 165)]
    camera = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

# Sub Function
def callback(data):
      capture(data.data)

def init_node():
      rospy.Subscriber("chatter", String, callback)
      rospy.spin() #listening loop

#main
if __name__=='__main__':
    # Camera Threading
    my_thread = threading.Thread(target = cameara)
    my_thread.start()
    # bridge
    lane_init()
    init_node()



