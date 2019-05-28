import numpy as np
import os
import random
from sklearn.model_selection import train_test_split


def load_data(data_path, num_nodes=8):
    '''
    Load text files in <data_path> and return numpy array of stacked sensor data.
    Output np array has shape of (TIME, NUM_NODES, 9).
    Last element of shape, that is 9, represents label of each frame and 8bit sensor data.
    Each label is digitized, which means, 0(0 person), 1(1-3 people), 2(4-6 people), 3(more than 7 people).
    
    For example, there are 8 nodes in 301 building and each node has 160000-320000 rows of 8bit sensor data.
    Then, output np array will have shape of (160000, 8, 9)
    
    
    :param: data_path string which represents path of sensor data
    :param: num_nodes integer which represents number of nodes
    :return data numpy array of shape (TIME, NUM_NODES, 9)
    '''
    
    data = []
    for i in range(num_nodes):
        curr = np.loadtxt(os.path.join(data_path, 'sensorData_{}.txt'.format(i+1)), delimiter=',', dtype=np.float32)
        data.append(curr)

    # data.shape: (time, node, 9)    
    data = np.stack(data, axis=1)
    
    # apply label
    bins=[1, 4, 7]
    data[:, :, 0] = np.digitize(data[:, :, 0], bins)
    
    return data

def prepare_data(data_path, num_nodes, num_frame, after):
    '''
    prediction task에서는 sensor data window를 사용하므로 window를 고려한 train, valid, test을 준비해야함
    GCN을 이용할 것이므로 각각의 dataset은 서로 시간이 겹치면 안됨
    또한 label이 매우 불균형 하므로 label을 기준으로 under-sampling을 진행한 후 train을 진행함
    
    - load_data 함수를 이용하여 data를 load함
    - data: (num_frame, num_nodes, 9)를 x: (num_frame, num_nodes, 8), y: (num_frame, num_nodes)로 분리함
    - window를 적용하여 x: (num_frame', num_nodes, 8), y: (num_frame', num_nodes)로 변경
    - time을 기준으로 train, valid, test set으로 분리
    - 각각의 dataset에서 label 숫자를 기준으로 (time, node_number)를 골라냄
    - 골라낸 (time, node_number)를 time 기준으로 group화하여 time 마다 길이가 num_nodes인 mask vector를 만듬
    
    :param: data_path string which denotes raw sensor data path
    :param: num_nodes integer which denotes number of nodes
    :param: num_frame integer which denotes window size and this should be multiple of 8
    :param: after integer which denotes frame difference between window and label and this also should be multiple of 8
    :return ((train_x, train_y, tarin_mask), (valid_x, valid_y, valid_mask), (test_x, test_y, test_mask))
    '''
    
    def get_mask(y):
        '''
        y: numpy array of shape (NUM_EXAMPLES, NUM_NODES) denotes label
        return: numpy array whose row in [t, n1, ... nn]
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
            # random.seed(0)
            random.shuffle(lst)
            result.extend(lst[:upto])

        # labels in original set
        for i in range(4):
            print('label %d: %d' % (i, len(arr[i])))
                        
        # until now, result consists of (t, n)
        # group by t
        dic = dict()
        for (t, n) in result:
            if t in dic:
                dic[t].append(n)
            else:
                dic[t] = [n]

        # convert to [t, n1, ... nn]
        mask = []
        for t in dic:
            row = [t] + [0] * num_nodes
            for n in dic[t]:
                row[n+1] = 1
                
            mask.append(row)
            
        return np.sort(np.array(mask), axis=0)
    
    
    # load data
    data = load_data(data_path) # shape: (TIME, NUM_NODES, 9)
    
    # split into x, y and apply window
    # this should be fixed to 'numpytic way' instead of for loop
    # cf) python append(list append) is faster than np.append => https://stackoverflow.com/questions/29839350/numpy-append-vs-python-append
    x, y = [], []
    for t in range(data.shape[0]-num_frame-after):
        curr_x = data[t:t+num_frame, :, 1:] # shape: (NUM_FRAME, NUM_NODES, 8)
        curr_x = np.swapaxes(curr_x, 0, 1) # shape: (NUM_NODES, NUM_FRAME, 8)
        curr_y = data[t+num_frame+after-1, :, 0] # shape: (NUM_NODES)
        x.append(curr_x.tolist())
        y.append(curr_y.tolist())
        
    x = np.array(x) # shape: (NUM_EXAMPLES, NUM_NODES, NUM_FRAME, 8)
    y = np.array(y) # shape: (NUM_EXAMPLES, NUM_NODES)
    
    # split into train/valid/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    # extract (t, n)s proportional to labels
    mask_train = get_mask(y_train)
    mask_valid = get_mask(y_valid)
    mask_test = get_mask(y_test)
    
    return (x_train, y_train, mask_train), (x_valid, y_valid, mask_valid), (x_test, y_test, mask_test)

def get_window(data_path, frame_size, after, num_nodes=8):
    '''
    given frame_size(window_size) and after, return x(feature) and y(label) from raw sensor data
    of shape: (time, num_nodes, 9)
    
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

def pick_index(y):
    '''
    get_window 함수를 통해 data_path에 존재하는 sensor data을 read한 후
    num_examples = total_frames - frame_size - after + 1인
    x: (num_examples, num_nodes, frame_size, 8)
    y: (num_examples, num_nodes)
    를 얻을 수 있다
    
    y에 존재하는 label은 매우 0, 1, 2, 3으로 이루어진 매우 불균형한 data이다.
    따라서 test set을 통해 performace를 측정할 때 주어진 test set의 모든 label을 사용하지 않고
    각 label별로 균등하게 뽑은 label에 대해서만 test performance를 측정한다.
    
    :param y numpy array of shape: (num_examples, num_nodes, num_classes), onehot encoded
    
    :return
    2d numpy array of [example_index, node_index]s
    '''

    arr = [[], [], [], []]
    y = np.argmax(y, axis=2) # shape: (num_examples, num_nodes)
    y = y.tolist()
    for t in range(len(y)):
        for n in range(len(y[0])):
            label = y[t][n]
            arr[label].append((t, n))
            
#     min_count = min([len(x) for x in arr])
    min_count = 500
    result = []
    for lst in arr:
        random.seed(0)
        random.shuffle(lst)
        result.extend(lst[:min_count])
    
    
    return np.array(result)
