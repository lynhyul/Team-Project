import numpy as np

#Data Load
data_size = 1643
y_data = []
# Read Value 00
f = open('./learning_data_0/textfile.txt', 'r')
lines = f.readlines() #각 줄을 요소로 같은 list
cnt = 0
for line in lines:
    cnt += 1
    y_data.append(line.split())
    if cnt%100 == 0:
        print('Data Load: {0}'.format(cnt),end = '\r')
    if cnt == data_size:
        break
#remove index
y_data = np.array(y_data)[:,1:]
#str -> float
y_data = y_data.astype('float32')
y_data = np.squeeze(y_data)
print(y_data.shape)
for inx, pos in enumerate(y_data):
    # Steering Calc
    # Instant
    Hub = 77.
    k_p = 0.01
    r_mid_pos = 80
    # Variables
    steering = 0
    steering_ang = 0

    # Calculate steering
    steering = k_p*(pos - r_mid_pos)
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

    '''steering_ang = steering_ang - 60
    steering_ang = steering_ang//3
    print(steering_ang)'''

    # save
    file_a = open('/home/autocar/catkin_ws/src/msg_ss/scripts/Learning/learning_data_0/SteerLog.txt','a')
    file_a.write('%d %f\n'%(inx+1, steering_ang))