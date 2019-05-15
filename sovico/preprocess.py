#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import os
from tqdm import tqdm
import random


# In[ ]:


def load_data(data_path, num_nodes=8):
    '''
    read sensor data from given data path
    
    :param data_path the path of data, e.g. sensorData_<node_num>.txt
    :param num_nodes number of nodes in data, in case of 301 building, 8 nodes exist
    
    :return
    numpy array of shape (time, num_nodes, 9)
    time: total number of frame
    num_nides: total number of nodes
    9: label at index [0] and 8bit sensor data at slice [1:9]
    '''
    
    print('start "load_data(%s, %d)"...' % (data_path, num_nodes))
    data = []
    for i in tqdm(range(num_nodes), desc='Nodes'):
        curr = np.loadtxt(os.path.join(data_path, 'sensorData_{}.txt'.format(i+1)), delimiter=',', dtype=np.float32)
        data.append(curr)

    # data.shape: (time, node, 9)    
    data = np.stack(data, axis=1)
    
    # apply label
    bins=[1, 4, 7]
    data[:, :, 0] = np.digitize(data[:, :, 0], bins)

    print('end "load_data(%s, %d)"!' % (data_path, num_nodes))
    return data


# In[1]:


def get_window(data_path, frame_size, after, num_nodes=8):
    '''
    given frame_size(window_size) and after, return x(feature) and y(label) from raw sensor data
    of shape: (time, num_nodes, 9)
    python append(list append) is faster than np.append => https://stackoverflow.com/questions/29839350/numpy-append-vs-python-append
    
    :param data_path the path of data, e.g. sensorData_<node_num>.txt
    :param frame_size number of frame to use at prediction, that is window size
    :param after time difference between last time of input feature and target time
    :param num_nodes number of nodes in data, in case of 301 building, 8 nodes exist
    
    :return
    tuple of (x, y)
    x: numpy array of shape (num_examples, num_nodes, frame_size, 8)
    num_examples = time(total number of frames) - frame_size - after + 1
    8 represents our sensor data has 8bit information
    y: numpy array of shape (num_examples, num_nodes) which represents label of each position and time
    label 0 => 0 person, label 1 => 1~3 people, label 2 => 4~6 people, label 3 => more than 7 people
    '''
    
    print('start "get_window(%s, %d, %d, %d)"...' % (data_path, frame_size, after, num_nodes))
    data = load_data(data_path) # shape: (time, num_nodes, 9)
    num_examples = data.shape[0] - frame_size - after + 1
    x = [] # use python list for append performace
    y = [] # use python list for append performace
    
    for t in tqdm(range(num_examples), desc='Total frame'):
        curr_x = data[t:t+frame_size, :, 1:] # shape: (frame_size, num_nodes, 8)
        curr_y = data[t+frame_size+after-1, :, 0] # shape: (num_nodes)
        curr_x = np.swapaxes(curr_x, 0, 1) # shape: (num_nodes, frame_size, 8)
        
        x.append(curr_x.tolist())
        y.append(curr_y.tolist())
    
    print('end "get_window(%s, %d, %d, %d)"!' % (data_path, frame_size, after, num_nodes))
    return np.array(x), np.array(y)


# In[31]:


def pick_index(data_path, frame_size, after, num_nodes=8):
    '''
    주어진 sensor data는 shape가 (N, number of frame, 9)인 3차원 tensor이다.
    N은 number of nodes를 의미한다.
    number of frame은 총 frame의 수를 의미하며 각 frame은 1/8초 간격으로 생성(측정)된다.
    마지막 9는 sensor data와 사람의 숫자를 의미한다. 0번 index가 사람의 숫자이며 1~8번 index는 8bit binary sensor data를
    나타낸다.
    
    우리의 task는 정확한 사람의 숫자를 예측하는 것이 아니라 0(0명), 1(1~3명), 2(4~6명), 3(7명 이상)의 사람 수의 카테고리를
    예측하는 것이다. simulation으로 생성된 data를 살펴본 바 대부분의 label이 0으로 되어있음을 확인하였다. 그 결과 label간의 균형을
    맞추어 학습을 진행하기 위해 전체 data를 그대로 예측하지 않고 label을 기준으로 균등하게 선택된 example에 대해서만 예측을 진행한다.
    label은 hyper parameter인 (frame_size, after)에 의해 결정된다. 특히 sensor를 설치하는 graph가 바뀌는 경우를 대비하여
    num_nodes도 pick_index function의 parameter로 이용한다.
    
    균등하게 선택된 예측하고자 하는 label은 (time, node_number)로 표현할 수 있다.
    즉 input feature는 data[node_number, time-after-frame_size:time-after, 1:9],
    output label은 data[node_number, time, 0]으로 나타낼 수 있다.
    
    실험하고자 하는 model은 FNN, RNN, GCN 등인데 FNN과 RNN은 특정 시점 t의 특정 node n만을 이용하여 실험을 진행할 수 있지만
    GCN은 전체 graph 정보를 다 이용한다. 그래서 FNN/RNN과 GCN이 똑같은 (t, n)에 대해 예측을 진행할 수 있도록 pick_index
    를 통해 예측하고자 하는 label 정보를 return해 준다.
    
    return: list of (t, n), t는 t번째 frame, n은 n번째 node를 의미함
    0 <= t < total_frame
    0 <= n < num_nodes
    '''
    result = [[], [], [], []] # result[k] => [t, n] with label k
    threshold = 5000 # minimum number of selected examples per label
    seed = 5 # random seed
    print('start "pick_index"...')
    
    data = load_data(data_path, num_nodes)
    # iter over time
    for t in range(data.shape[0]):            
        if t-after-frame_size+1 < 0:
            continue
            
        for n in range(num_nodes):
            label = data[t, n, 0]
            result[label].append((t, n))
            
    # shuffle
    random.seed(seed)
    for i in range(4):
        random.shuffle(result[i])

    print('end "pick_index"!')
    return [
        *result[0][:threshold],
        *result[1][:threshold],
        *result[2][:threshold],
        *result[3][:threshold],
    ]


# In[ ]:




