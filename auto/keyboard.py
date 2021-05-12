import xhat as hw
import time
import cv2
import config as cfg

import os
import sys
import signal
import csv

def recording():
    if cfg.recording:
        cfg.recording = False
        cfg.f.close()
    else:
        cfg.recording = True
        if cfg.currentDir == '':
            cfg.currentDir = time.strftime('%Y-%m-%d')
            os.mkdir(cfg.outputDir+cfg.currentDir)
            cfg.f=open(cfg.outputDir+cfg.currentDir+'/data.csv','w')
        else:
            cfg.f=open(cfg.outputDir+cfg.currentDir+'/data.csv','a')
        cfg.fwriter = csv.writer(cfg.f)

def saveimage():
    if cfg.recording:
        myfile = 'img_'+time.strftime('%Y-%m-%d_%H-%M-%S')+'_'+str(cfg.cnt)+'.jpg'
        print(myfile, cfg.wheel)

        cfg.fwriter.writerow((myfile, cfg.wheel))

        cv2.imwrite(cfg.outputDir+cfg.currentDir+'/'+ myfile,full_image)

        cfg.cnt += 1

        


if __name__ == '__main__':

    start_flag = False
    
    c = cv2.VideoCapture(0)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    #c.set(cv2.CAP_PROP_FPS, 15)


    while(True):
        _,full_image = c.read()
        
        cv2.imshow('frame',full_image)
    
        k = cv2.waitKey(5)
        if k == ord('q'):  #'q' key to stop program
            break

        """ Toggle Start/Stop motor movement """
        if k == 115: #115:'s'
            if start_flag == False: 
                start_flag = True
            else:
                start_flag = False
            print('start flag:',start_flag)

        """ Toggle Record On/Off  """
        if k == 114: #114:'r'
            recording()
            if cfg.recording:
               start_flag = True
            else:
               start_flag = False
               cfg.cnt = 0
            print('cfg.recording:',cfg.recording)

        #save image files and images list file   
        if cfg.recording:
            saveimage()
            print(cfg.cnt)
        
        if start_flag:
            # Left arrow: 81, Right arrow: 83, Up arrow: 82, Down arrow: 84
            if k == 81: 
                hw.motor_one_speed(cfg.maxturn_speed)
                hw.motor_two_speed(cfg.minturn_speed)
                #print('Straight')
                cfg.wheel = 1
            if k == 83: 
                hw.motor_one_speed(cfg.minturn_speed)
                hw.motor_two_speed(cfg.maxturn_speed)
                cfg.wheel = 3
            if k == 82: 
                hw.motor_one_speed(cfg.normal_speed_right)
                hw.motor_two_speed(cfg.normal_speed_left)
                cfg.wheel = 2
        
        else:
            hw.motor_one_speed(0)
            hw.motor_two_speed(0)
            cfg.wheel = 0

        
hw.motor_clean()
cv2.destroyAllWindows()
