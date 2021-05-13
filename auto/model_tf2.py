import tensorflow as tf
import numpy as np
###import scipy


stddev = 0.1

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,GaussianNoise, Dropout

input = Input(shape=x_train.shape[1:])  # image_shape =[None, 66, 200, 3]
gaus= GaussianNoise(stddev) (input) # stddev = 노이즈 분포의 부동, 표준 편차
conv1 = Conv2D(24, (5, 5), activation='relu', padding='valid',stride=2)(gaus)
conv2 = Conv2D(36, (5, 5), activation='relu', padding='valid',stride=2)(conv1)
conv3 = Conv2D(48, (5, 5), activation='relu', padding='valid',stride=2)(conv2)
conv4 = Conv2D(64, (3, 3), activation='relu', padding='valid',stride=1)(conv3)
conv5 = Conv2D(64, (3, 3), activation='relu', padding='valid',stride=1)(conv4)
flatten = Flatten() (conv5)
fc1 = Dense(100, activation = 'relu') (flatten)
fc1_drop = Dropout(0.1) (fc1)
fc2 = Dense(50, activation = 'relu') (fc1_drop)
fc2_drop = Dropout(0.1) (fc2)
fc3 = Dense(10, activation = 'relu') (fc2_drop)
fc3_drop = Dropout(0.1) (fc3)
output = Dense(4, activation = 'softmax') (fc3_drop)

model = Model(inputs= input, outputs=output)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

