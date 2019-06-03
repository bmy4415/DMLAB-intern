#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import numpy as np
import os
import random
import pickle
import logging
from sklearn.model_selection import train_test_split


# In[ ]:


def initialize_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # console logging
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # file logging
    handler = logging.FileHandler(os.path.join(log_dir, 'log'), 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# In[ ]:


def load_data(data_path, NUM_NODES=8):
    '''
    Load text files in <data_path> and return numpy array of stacked sensor data.
    Output np array has shape of (TOTAL_FRAMES, NUM_NODES, 9).
    Last element of shape, that is 9, represents label of each frame(1bit) and sensor data(8bit).
    Each label is digitized, which means, 0(0 person), 1(1-3 people), 2(4-6 people), 3(more than 7 people).
    
    For example, there are 8 nodes in 301 building and each node has 160000-320000 rows of 8bit sensor data.
    Then, output np array will have shape of (160000, 8, 9)
    
    :param: data_path string which represents path of sensor data
    :param: NUM_NODES integer which represents number of nodes
    :return data numpy array of shape (TOTAL_FRAMES, NUM_NODES, 9)
    '''
    
    data = []
    for i in range(NUM_NODES):
        curr = np.loadtxt(os.path.join(data_path, 'sensorData_{}.txt'.format(i+1)), delimiter=',', dtype=np.float32)
        data.append(curr)

    # data.shape: (TOTAL_FRAMES, NUM_NODES, 9)    
    data = np.stack(data, axis=1)
    
    # apply label
    bins=[1, 4, 7]
    data[:, :, 0] = np.digitize(data[:, :, 0], bins)

    return data


# In[ ]:


def prepare_data(data_path, window_size, after, NUM_NODES=8):
    '''
    prediction task에서는 sensor data의 window를 사용하므로 window를 고려한 train, valid, test을 준비해야함
    GCN을 이용할 것이므로 각각의 dataset은 서로 시간이 겹치면 안됨
    또한 label이 매우 불균형 하므로 label을 기준으로 under-sampling을 진행한 후 train을 진행함
    
    - load_data 함수를 이용하여 data를 load함
    - data: (TOTAL_FRAMES, NUM_NODES, 9)를 x: (TOTAL_FRAMES, NUM_NODES, 8), y: (TOTAL_FRAMES, num_nodes)로 분리함
    - window를 적용하여 x: (NUM_EXAMPLES, NUM_NODES, 8), y: (NUM_EXAMPLES, NUM_NODES)로 변경
    - time 기준으로 train, valid, test set으로 분리
    - 각각의 dataset에서 label 숫자를 기준으로 (time, node_number)를 골라냄
    - 골라낸 (time, node_number)를 time 기준으로 group화하여 time 마다 길이가 num_nodes인 mask vector를 만듬
    
    :param: data_path string which denotes raw sensor data path
    :param: window_size integer which denotes window size and this should be multiple of 8
    :param: after integer which denotes frame difference between window and label and this also should be multiple of 8
    :param: NUM_NODES integer which denotes number of nodes
    :return ((train_x, train_y, tarin_mask), (valid_x, valid_y, valid_mask), (test_x, test_y, test_mask))
    '''
    
    def get_mask(y):
        '''
        return mask for prediction task
        mask is 2d array and lenght of row is NUM_NODES+1
        mask의 row의 첫번째 값은 time값이고 그 이후의 N개의 값은 해당 time에 있는 N개의 node 중에서 각각의 node를 loss를 계산할 때
        이용할 것인지에 대한 0,1 값을 나타냄
        예를들어 row: [126, 0, 1, 0, 0, 1] 인 경우, time=126인 시점에서 1, 4번 node만 loss 계산에 사용하겠다는 것을 의미함
        
        remind
        이러한 mask를 이용하는 이유는 GCN의 input에 특정 time의 모든 node를 다 넣어야하고 동시에 똑같은 input을
        FNN과 RNN에도 적용할 수 있어야 하기 때문임
        
        :param: y numpy array of shape (NUM_EXAMPLES, NUM_NODES) which denotes label of each sensor and this is not one-hot encoded
        :return numpy 2d array of shape: (NUM_EXAMPLES, NUM_NODES+1) each row denotes [time, Node1, ... NodeN]
        '''
        
        arr = [[], [], [], []]
        y = y.tolist()
        for t in range(len(y)):
            for n in range(len(y[0])):
                label = int(y[t][n])
                arr[label].append((t, n))

        # upto = min([len(x) for x in arr])
        upto = 500
        result = []
        for lst in arr:
            random.seed(0)
            random.shuffle(lst)
            result.extend(lst[:upto])

        # labels in original set
        # for i in range(4):
        #     print('label %d: %d' % (i, len(arr[i])))
                        
        # until now, result consists of (time, node_number)
        # group by time
        dic = dict()
        for (t, n) in result:
            if t in dic:
                dic[t].append(n)
            else:
                dic[t] = [n]

        # convert to row format: [t, n1, ... nn]
        mask = []
        for t in dic:
            row = [t] + [0] * NUM_NODES
            for n in dic[t]:
                row[n+1] = 1
                
            mask.append(row)
              
        mask = np.array(mask)
        mask = mask[mask[:, 0].argsort()] # sort by time asc
        return mask
    
    
    # load data
    data = load_data(data_path) # shape: (TIME, NUM_NODES, 9)
    
    # split into x, y and apply window
    # this should be fixed to 'numpytic way' instead of for loop
    x, y = [], []
    for t in range(data.shape[0]-window_size-after):
        curr_x = data[t:t+window_size, :, 1:] # shape: (window_size, NUM_NODES, 8)
        curr_x = np.swapaxes(curr_x, 0, 1) # shape: (NUM_NODES, window_size, 8)
        curr_y = data[t+window_size+after-1, :, 0] # shape: (NUM_NODES)
        x.append(curr_x.tolist())
        y.append(curr_y.tolist())
        
    x = np.array(x) # shape: (NUM_EXAMPLES, NUM_NODES, window_size, 8)
    y = np.array(y) # shape: (NUM_EXAMPLES, NUM_NODES)
    
    # split into train/valid/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    # extract (t, n)s proportional to labels
    mask_train = get_mask(y_train)
    mask_valid = get_mask(y_valid)
    mask_test = get_mask(y_test)
    
    return (x_train, y_train, mask_train), (x_valid, y_valid, mask_valid), (x_test, y_test, mask_test)


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='sensor data path')
    parser.add_argument('--window_size', type=int, required=True, help='number of frame in window')
    parser.add_argument('--after', type=int, required=True, help='number of after window')
    parser.add_argument('--save_path', type=str, required=True, help='save data path')

    args = parser.parse_args()

    # main function
    filename = 'prediction_preprocessed_{}window_{}after.pkl'.format(args.window_size, args.after)
    path = os.path.join(args.save_path, filename)
    
    if os.path.exists(path):
        print(path, 'already exists!')
        
    else:
        print('start preprocess...')
        
        preprocessed = prepare_data(
            data_path=args.data_path,
            window_size=args.window_size,
            after=args.after,
            NUM_NODES=8
        )
        
        # save as pickle file
        with open(path, 'wb') as f:
            pickle.dump(preprocessed, f)
            
        print(path, 'saved!')


# In[ ]:




