#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import threading
import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String, Int32, Float32
import tensorflow as tf
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
global sess
global m1
global pub
global camera

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

#NN Model
class Model:
    def __init__(self, sess, name):  #생성자
        self.sess = sess 
        self.name = name #모델을 생성할때 스코프로 묶을 때 사용할 이름
        self._set_param() #변수 입력 함수
        self._build_net() #신경망 구성 함수
        self.saver = tf.train.Saver()
    def _set_param(self):
        self._learning_rate = tf.placeholder(tf.float32)
        self._keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, [None, 180, 320, 3])
        self.Y = tf.placeholder(tf.float32, [None, 26])
        self.X_img = tf.reshape(self.X,[-1,180,320,3])

    def _build_net(self): #생성자에서 기본 실행, 신경망 구성
        with tf.variable_scope(self.name):
            #CNN Layer 01
            self.L1 = tf.layers.conv2d(inputs=self.X_img, filters=64, kernel_size= [3,3], padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.L1 = tf.nn.max_pool(self.L1,ksize=[1,2,2,1],strides=[1,2,2,1],
                                     padding='SAME')
            self.L1 = tf.nn.dropout(self.L1, keep_prob=self._keep_prob)
            #CNN Layer 02
            self.L2 = tf.layers.conv2d(inputs=self.L1, filters=64, kernel_size= [3,3], padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.L2 = tf.nn.max_pool(self.L2,ksize=[1,2,2,1],strides=[1,2,2,1],
                                     padding='SAME')
            self.L2 = tf.nn.dropout(self.L2, keep_prob=self._keep_prob)
            #CNN Layer 03
            self.L3 = tf.layers.conv2d(inputs=self.L2, filters=128, kernel_size= [3,3], padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.L3 = tf.nn.max_pool(self.L3,ksize=[1,2,2,1],strides=[1,2,2,1],
                                     padding='SAME')
            self.L3 = tf.nn.dropout(self.L3,keep_prob=self._keep_prob)
            #self.FC_input = tf.reshape(self.L3,[-1,8*8*128])
            self.FC_input = tf.layers.flatten(self.L3)

            #FC Layer 01
            self.FC = tf.layers.dense(inputs=self.FC_input, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.FC = tf.nn.dropout(self.FC,keep_prob=self._keep_prob)
            #FC Layer 02
            self.FC = tf.layers.dense(inputs=self.FC, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.FC = tf.nn.dropout(self.FC,keep_prob=self._keep_prob)
            #FC Layer 03
            self.FC = tf.layers.dense(inputs=self.FC, units=256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.FC = tf.nn.dropout(self.FC,keep_prob=self._keep_prob)
            #FC Layer 04, no dropout
            self.FC = tf.layers.dense(inputs=self.FC, units=26, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.hyp = tf.compat.v1.nn.softmax(self.FC)
            
            #cost,train tensor 인데, 이것도 tensorflow 개념에선 node라서, 여기에 해줌.
            #self.cost = tf.reduce_mean(tf.square(self.hyp - self.Y))
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= self.FC, labels= self.Y))
            self._train = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

            numbering_h = tf.argmax(self.hyp,1) #one-hot -> 번호(1,2,3...)로 변환
            numbering_Y = tf.argmax(self.Y,1) #Y는 매트릭스고, 이건 벡터다.
            compare_result = tf.cast(tf.equal(numbering_h,numbering_Y),tf.float32)
            self.accuracy = tf.reduce_mean(compare_result)

    def train(self, x_data, y_data, learning_rate=0.0001, keep_prob=0.7):
        feed_train = {self.X:x_data, self.Y:y_data, self._learning_rate:learning_rate, self._keep_prob:keep_prob}
        return self.sess.run([self.cost,self._train], feed_dict=feed_train)

    def predict(self, x_data, y_data):
        feed_pred = {self.X:x_data, self.Y:y_data, self._keep_prob:1}
        return self.sess.run(self.cost, feed_dict=feed_pred)

    def operator(self, x_data):
        feed_pred = {self.X:x_data, self._keep_prob:1}
        return self.sess.run(self.hyp, feed_dict=feed_pred)

    def get_accuracy(self, x_data, y_data):
        feed_pred = {self.X:x_data, self.Y:y_data, self._keep_prob:1}
        return self.sess.run(self.accuracy, feed_dict=feed_pred)


# Thread camera Function
def cameara():
    global img

    while(True):
        while(True):
            ret, img = cap.read()
            if(ret==True):
                break
        #print("is capturing")
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
    '''last_mid_pos = 0
    file_mid_log = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/mid_line_log/mid_log.txt','r')
    while True:
        line = file_mid_log.readline()
        if not line: break
        last_mid_pos = int(line)
    #print("last mid-line position: {0}".format(last_mid_pos))
    file_mid_log.close()'''

    # image resize
    test_img = cv2.resize(img, dsize=(320, 180), interpolation=cv2.INTER_AREA)
    test_img_copy = np.array(test_img).astype(np.uint8)
    '''
    # Undistort and transform to overhead view
    img_dash_undistorted = camera.undistort(test_img_copy)
    img_overhead = camera.warp_to_overhead(img_dash_undistorted)'''
    
    # find lane with NN
    mid_line_onehot = m1.operator([test_img_copy])
    mid_line_imm = np.argmax(mid_line_onehot,1)
    steering_ang = (mid_line_imm * 3) + 32
    

    # publish data
    pub.publish(steering_ang)
    #print("send mid_line: {0}".format(mid_line))

    '''
    # log mid line
    file_a = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/textfile.txt','a')
    file_a.write('%d %f\n'%(data, mid_line))

    file_mid_log = open('/home/autocar/catkin_ws/src/msg_ss/scripts/data/mid_line_log/mid_log.txt','w')
    file_mid_log.write('%d'%(mid_line))'''

    print("process time:{0}".format(time.time()-vector))

    # save img
    #cv2.imwrite("/home/autocar/camera_im/camera_image"+str(data)+".jpeg",test_img_copy)
    #lane_finder.viz_pipeline(test_img_copy)
    #plt.show()

def lane_init():
    global camera

    # Calibrate using checkerboard
    calibration_img_files = glob.glob('/home/autocar/catkin_ws/src/msg_ss/scripts/data/camera_cal_low/*.jpg')
    #lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]
    lane_shape = [(55, 80), (265, 80), (0, 165), (320, 165)]
    camera = DashboardCamera(calibration_img_files, chessboard_size=(9, 6), lane_shape=lane_shape)

def NN_init():
    global sess
    global m1

    #Save path
    save_file = '/home/autocar/catkin_ws/src/msg_ss/scripts/Learning/Torcs_NN'
    #session
    sess = tf.Session()
    #Graphs
    m1 = Model(sess,'m1')
    #NN load
    m1.saver.restore(sess, save_file)
    print("NN init completed")

# Sub Function
def callback(data):
      capture(data.data)

def init_node():
    print("init node")
    rospy.Subscriber("chatter", String, callback)
    rospy.spin() #listening loop

#main
if __name__=='__main__':
    # Camera Threading
    my_thread = threading.Thread(target = cameara)
    my_thread.start()
    # bridge
    lane_init()
    NN_init()
    init_node()



