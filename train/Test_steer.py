import tensorflow as tf
import PIL.Image as pilimg
import numpy as np
import glob
import matplotlib.pyplot as plt
from natsort import natsorted

#입력: 64x64 이미지
#출력: 2노드 출력

#one hot
def one_hot(input, numcls):
    #return np.eye(numcls)[input-1,:]
    return np.identity(numcls)[input]

#입력: 64x64 이미지
#출력: 2노드 출력

#Data Load
data_size = 1600
x_data = []
y_data = []
y_onehot = []
# Read image 00
print("<Data load set 00>")
img_list = glob.glob('./learning_data_0/camera_im/*.jpeg')
img_list_sorted = natsorted(img_list,reverse=False)  # Sort the images

cnt = 0
for img_file in img_list_sorted:
    cnt += 1
    img = plt.imread(img_file)
    im = np.array(img)
    x_data.append(im)
    if cnt%10 == 0:
        print('Image Load: {}/{}'.format(cnt, data_size),end = '\r')
    if cnt == data_size:
        break
'''
# Read image 01
print("<Data load set 01>")
img_list = glob.glob('./learning_data_1/camera_im/*.jpeg')
img_list_sorted = natsorted(img_list,reverse=False)  # Sort the images

cnt = 0
for img_file in img_list_sorted:
    cnt += 1
    img = plt.imread(img_file)
    im = np.array(img)
    x_data.append(im)
    if cnt%10 == 0:
        print('Image Load: {}/{}'.format(cnt, data_size),end = '\r')
    if cnt == data_size:
        break
'''

#list to array
x_data = np.array(x_data)
#unint8 to int32
x_data = x_data.astype('int32')
print('Image data: {}'.format(len(x_data)))


# Read Value 00
f = open('./learning_data_0/SteerLog.txt', 'r')
lines = f.readlines() #각 줄을 요소로 같은 list
cnt = 0
for line in lines:
    cnt += 1
    y_data.append(line.split())
    if cnt%100 == 0:
        print('Data Load: {0}'.format(cnt),end = '\r')
    if cnt == data_size:
        break
'''
# Read Value 01
f = open('./learning_data_1/textfile.txt', 'r')
lines = f.readlines() #각 줄을 요소로 같은 list
cnt = 0
for line in lines:
    cnt += 1
    y_data.append(line.split())
    if cnt%100 == 0:
        print('Data Load: {0}'.format(cnt),end = '\r')
    if cnt == data_size:
        break
'''
#remove index
y_data = np.array(y_data)[:,1:]
#str -> float
y_data = y_data.astype('float32')

print('Position data: {}'.format(len(y_data)))

# set onehot
for steering_ang in y_data:
    tmp = steering_ang - 60
    tmp = tmp//3
    y_onehot.append(one_hot(tmp,21)) # ex) 0 -> [1,0,0...], 1 -> [0,1,0...]

#scaling
'''x_data = x_data/255
y_data[:,0] = y_data[:,0]*1
y_data[:,1] = y_data[:,1]*10
print('x_data:{0}, y_data:{1}'.format(x_data.shape,y_data.shape))
print(x_data[1,:,:])'''

#data separate
train_size = int(len(x_data) * 0.7)
test_size = len(x_data) - train_size

x_train = np.array(x_data[0:train_size])
y_train = np.array(y_onehot[0:train_size])

x_test = np.array(x_data[train_size:len(x_data)])
y_test = np.array(y_onehot[train_size:len(x_data)])

print('x_train:{0}, y_train:{1}'.format(x_train.shape,y_train.shape))
print('x_test:{0}, y_test:{1}'.format(x_test.shape,y_test.shape))


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
        self.Y = tf.placeholder(tf.float32, [None, 21])
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
            self.FC = tf.layers.dense(inputs=self.FC, units=21, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.hyp = tf.nn.softmax(self.FC)
            
            #cost,train tensor 인데, 이것도 tensorflow 개념에선 node라서, 여기에 해줌.
            #self.cost = tf.reduce_mean(tf.square(self.hyp - self.Y))
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= self.FC, labels= self.Y))
            self._train = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

            numbering_h = tf.argmax(self.hyp,1) #one-hot -> 번호(1,2,3...)로 변환
            numbering_Y = tf.argmax(self.Y,1) #Y는 매트릭스고, 이건 벡터다.
            compare_result = tf.cast(tf.equal(numbering_h,numbering_Y),tf.float32)
            self.accuracy = tf.reduce_mean(compare_result)

    def train(self, x_data, y_data, learning_rate=0.001, keep_prob=0.7):
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


#Save path
save_file = './Torcs_NN'
#session
sess = tf.Session()
#Graphs
m1 = Model(sess,'m1')
#NN load
m1.saver.restore(sess, save_file)
#test
'''batch_size = 100
global_step = 0
batch_num = int(len(x_test)/batch_size)
cost_avg = 0
for i in range(batch_num):
    global_step += 1
    batch_ptr = i*batch_size
    batch_x, batch_y = [x_test[batch_ptr : (batch_ptr + batch_size), :, :],
                        y_test[batch_ptr : (batch_ptr + batch_size), :]]
    cost_curr = m1.predict(batch_x, batch_y)
    cost_avg += cost_curr/batch_num
    print("step: ",global_step+1, "cost: ",cost_curr)'''
#operate
list_x = []
list_y = []
for cnt in range(0,3499):
    output = m1.operator([x_data[cnt]])
    x_numbered = np.argmax(output,1)
    y_numbered = np.argmax(y_onehot,1)
    #print("index {0}: output {1}, label {2}".format(cnt,x_numbered[0],y_numbered[cnt]))
    list_x.append(x_numbered[0])
    list_y.append(y_numbered[cnt])

fig = plt.figure()
ax = fig.gca()
ax.plot(range(3499),list_x, color = 'orange')
ax.plot(range(3499),list_y)
plt.show()

accuracy = m1.get_accuracy(x_test[0:100],y_test[0:100])
print('Test accuracy: {}%'.format(100*accuracy))
