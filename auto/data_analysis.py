import scipy.misc

import random
import csv
#from mlxtend.preprocessing import one_hot
import numpy as np
import config as cfg

xs = []
ys = []

wheel0 = 0
wheel1 = 0
wheel2 = 0
wheel3 = 0

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.csv
with open('data/' + cfg.currentDir + '/data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        #print(row[0], row[1])
        xs.append('data/' + cfg.currentDir + '/' + row[0])
        ys.append(int(row[1]))
        if int(row[1]) == 0:
            wheel0 += 1
        elif int(row[1]) == 1:
            wheel1 += 1
        elif int(row[1]) == 2:
            wheel2 += 1
        elif int(row[1]) == 3:
            wheel3 += 1

print('Total data counts: ', len(xs))
print('Stop data counts: ', wheel0, ', ratio(%):', ' %0.1f' % (wheel0/len(xs)*100))
print('Left data counts: ', wheel1, ', ratio(%):', ' %0.1f' % (wheel1/len(xs)*100))
print('strait data counts: ', wheel2, ', ratio(%):', ' %0.1f' % (wheel2/len(xs)*100))
print('Right data counts: ', wheel3, ', ratio(%):', ' %0.1f' % (wheel3/len(xs)*100))


###ys = one_hot(ys, num_labels=4, dtype='int')

#print(np.reshape(ys, -1))


#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

"""
train_xs = xs[:int(len(xs) * 1)]
train_ys = ys[:int(len(xs) * 1)]

val_xs = xs[-int(len(xs) * 1):]
val_ys = ys[-int(len(xs) * 1):]
"""

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[cfg.modelheight:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[cfg.modelheight:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
