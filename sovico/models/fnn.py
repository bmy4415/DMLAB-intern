#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, '..') # to import preprocess.py

import os
import argparse
import pickle
import logging
import numpy as np

from keras.models import Sequential
from keras.backend import tensorflow_backend as K
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

from preprocess import initialize_logger
from sklearn.metrics import classification_report, accuracy_score


# In[ ]:


def preprocess(x, y, mask):
    '''
    preprocess.prepate_data()의 결과를 FNN 직접 입력으로 넣을 수 있도록 처리해줌
    NUM_EXAMPLES는 window를 적용한 후의 전체 TIME의 갯수이고
    NUM_TIMES가 실제 train에 사용할 TIME의 갯수를 의미함
    NUM_EXAMPLES와 NUM_TIMES가 다른 이유는 전체 NUM_EXAMPLES 중에서 label에 균형을 맞추어
    일부 TIME의 일부 NODE에 대해서만 training을 진행하기 때문임
    
    FNN에서는 GCN과 달리 graph structure를 필요로하지 않으므로 mask를 이용하여 
    x: (NUM_EXAMPLES, WINDOW_SIZE*8)
    y: (NUM_EXAMPLES, NUM_CLASSES)
    형태로만 전처리를 진행하면 됨
    
    :param: x (NUM_EXAMPLES, NUM_NODES, WINDOW_SIZE, 8)
    :param: y (NUM_EXAMPLES, NUM_NODES
    :param: mask (NUM_TIMES, NUM_NODES+1)
    
    :return (NUM_TIMES, x', y', mask')
    x': numpy array of shape (NUM_EXAMPLES, WINDOW_SIZE*8) which is flattened over WINDOW_SIZE
    y': numpy array of shape (NUM_EXAMPLES, NUM_CLASSES) which is one-hotted
    '''
    
    x = np.reshape(x, [x.shape[0], x.shape[1], x.shape[2]*x.shape[3]])
    y = to_categorical(y) # one-hot encoding
    
    x_ = []
    y_ = []
    
    # this can be improved usnig 'numpytic' code
    for i in range(mask.shape[0]):
        time = mask[i, 0]
        nodes = mask[i, 1:]
        for j in range(nodes.shape[0]):
            if nodes[j] == 1:
                x_.append(x[time, j, :].tolist())
                y_.append(y[time, j, :].tolist())
    
    x_ = np.array(x_) # shape: (NUM_EXAMPLES', WINDOW_SIZE*8)
    y_ = np.array(y_) # shape: (NUM_EXAMPLES', NUM_CLASSES)
    return x_, y_


# In[ ]:


class FNN():
    '''
    6-layer FNN network
    '''
    
    def __init__(self, window_size, fnn_hiddens):
        '''
        FNN constructor
        Each value in window is feature (8bit * window_size)
        
        :param: window_size number of frames to use
        :param: fnn_hiddens list of length 6 which denotes neruons in each layer
        '''

        input_dim = window_size*8
        num_classes = 4
        
        model = Sequential()
        model.add(Dense(fnn_hiddens[0], input_dim=input_dim))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dense(fnn_hiddens[1]))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dense(fnn_hiddens[2]))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(fnn_hiddens[3]))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(fnn_hiddens[4]))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dense(fnn_hiddens[5]))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dense(num_classes)) # output layer
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        
    def fit(self, data, save_dir, learning_rate=0.001, epochs=500, patience=10):
        '''
        train using given dataset and hyper-parameters and then save model to save_dir
        because we use validation set, we will save model to save_dir everywhen there comes best validation loss
        also we use early stopping with default patience as 10
        (e.g. stop training if validation loss does not get better during consecutive 20 epochs)
        
        :param: data tuple of 3 dataset(train, valid, test) which is output of preprocess.prepare_data()
        :param: save_dir path where trained model and train results will be saved
        :param: learning_rate
        :param: epochs
        '''

        logging.info('start fit function')
        
        # preprocess data for FNN
        # x: (NUM_EXAMPLES, WINDOW_SIZE*8)
        # y: (NUM_EXAMPLES, NUM_CLASSES)
        train, valid, test = data
        x_train, y_train = preprocess(*train)
        x_valid, y_valid = preprocess(*valid)
        x_test, y_test = preprocess(*test)
        
        assert x_train.shape[0] == y_train.shape[0]
        assert x_valid.shape[0] == y_valid.shape[0]
        assert x_test.shape[0] == y_test.shape[0]

        es = EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32,
                       validation_data=(x_valid, y_valid), callbacks=[es],
                       verbose=1)
        
        true_valid = np.argmax(y_valid, axis=1)
        pred_valid = np.argmax(self.model.predict(x_valid), axis=1)
        true_test = np.argmax(y_test, axis=1)
        pred_test = np.argmax(self.model.predict(x_test), axis=1)
        
        acc_valid = accuracy_score(true_valid, pred_valid)
        acc_test = accuracy_score(true_test, pred_test)

        # valid set classification report
        logging.info('valid acc: {}'.format(acc_valid))
        logging.info(classification_report(true_valid, pred_valid))
        # test set classification report
        logging.info('test acc: {}'.format(acc_test))
        logging.info(classification_report(true_test, pred_test))
        
        # save model
        self.model.save(os.path.join(save_dir, 'model.h5'))
        
        # load model
        # from keras.models import load_model
        # model = load_model('my_model.h5')


# In[ ]:


def get_modelname(window_size, after, fnn_hiddens):
    '''
    model을 식별할 수 있는 이름을 지정해주는 함수
    '''
    
    return 'fnn_window{}_after{}_dims{}'.format(window_size, after, fnn_hiddens)


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='preprocessed data dir')
    parser.add_argument('--window_size', type=int, required=True, help='number of frame in window')
    parser.add_argument('--after', type=int, required=True, help='number of after window')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save train result')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--fnn_hiddens', type=int, nargs='+')
    args = parser.parse_args()

    # log와 ckpt가 저장될 directory 생성
    modelname = get_modelname(args.window_size, args.after, args.fnn_hiddens)
    path = os.path.join(args.save_dir, modelname)
    if not os.path.isdir(path):
        os.mkdir(path)
        
    # logger 초기화
    initialize_logger(path)
        
    # load data
    data_filename = 'prediction_preprocessed_{}window_{}after.pkl'.format(args.window_size, args.after)
    data_filename = os.path.join(args.data_dir, data_filename)
    with open(data_filename, 'rb') as f:
        data = pickle.load(f)

    model = FNN(args.window_size, args.fnn_hiddens)
    model.fit(data, save_dir=path, learning_rate=args.lr)
    


# In[ ]:




