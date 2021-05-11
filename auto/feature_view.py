import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import time
import csv
import numpy as np
import config as cfg
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

# img = cv2.imread('steering_wheel_image.jpg',0)
# rows,cols = img.shape

smoothed_angle = 0

xs = []
ys = []

with open(cfg.outputDir+cfg.currentDir+'/data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        #print(row[0], row[1])
        xs.append(row[0])
        ys.append(row[1])

print(xs[0])

flat2 = []

i = 0
correct_num = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread('data/' + cfg.currentDir + '/' + xs[i] , mode="RGB")
    image = scipy.misc.imresize(full_image[cfg.modelheight:], [66, 200]) / 255.0
    
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})

    image1 = model.h_conv1.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})
    flat1 = np.reshape(image1, [-1, 72912])*255
    flat1 = np.reshape(flat1.astype(np.uint8), (31,98,24))

    #print(flat1[:,:,0], flat2.shape, flat2.dtype)
    #print(full_image.shape, full_image.dtype)
    #print(flat2[0,2,0])

    image2 = model.h_conv2.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})
    flat2 = np.reshape(image2, [-1, 23688])*255
    flat2 = np.reshape(flat2.astype(np.uint8), (14,47,36))
      
    image3 = model.h_conv3.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})
    flat3 = np.reshape(image3, [-1, 5280])*255
    flat3 = np.reshape(flat3.astype(np.uint8), (5,22,48))
    
    image4 = model.h_conv4.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})
    flat4 = np.reshape(image4, [-1, 3840])*255
    flat4 = np.reshape(flat4.astype(np.uint8), (3,20,64))
    
    
    #for j in range(24):
    #    cv2.imshow("feature" + str(j), flat2[:,:,j])
    #cv2.imshow("feature" + str(0), flat2[:,:,0])
    #cv2.waitKey(0)

   
    break

    i += 1


#fig = plt.figure(1)

for j in range(24):
    plt.subplot(8,4,j+1)
    #plt.subplot(4,8,j+1)
    plt.imshow(flat1[:,:,j],cmap='gray')
    plt.axis('off')

plt.show()

for j in range(36):
    plt.subplot(6,6,j+1)
    plt.imshow(flat2[:,:,j],cmap='gray')
    plt.axis('off')

plt.show()

for j in range(48):
    plt.subplot(8,6,j+1)
    plt.imshow(flat3[:,:,j],cmap='gray')
    plt.axis('off')

plt.show()

"""
for j in range(64):
    plt.subplot(8,8,j+1)
    plt.imshow(flat4[:,:,j],cmap='gray')
    plt.axis('off')

plt.show()
"""

#cv2.destroyAllWindows()

