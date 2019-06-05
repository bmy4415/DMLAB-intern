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
import tensorflow as tf
from keras.utils import to_categorical
from preprocess import initialize_logger
from sklearn.metrics import accuracy_score, f1_score, classification_report


# In[ ]:


def get_adj_matrix():
    """
    Return current graph topology.
    Adjacent matrix will be used at GCN later

    :return: current graph topology of 301 building 5th floor.
    ex) return =
    [[0, 1, 0, 0, 0, 0, 0, 1],
     [1, 0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0, 1, 0],
     [0, 0, 0, 0, 0, 1, 0, 1],
     [1, 0, 0, 0, 0, 0, 1, 0]]
    """

    arr = [[0]*8 for x in range(8)]
    for i in range(8):
        before, after = (i-1)%8, (i+1)%8
        arr[i][before] = 1
        arr[i][after] = 1
        
    return arr


# In[ ]:


def normalize_matrix(adj_matrix):
    '''
    return numpy array which represents normalized matrix, D^(-1/2) A D^(-1/2)
    '''
    ahat = np.array(adj_matrix) + np.eye(len(adj_matrix))
    dhat_prime = np.diag(1 / np.sqrt(np.sum(ahat, axis=1)))
    return np.matmul(np.matmul(dhat_prime, ahat), dhat_prime)


# In[ ]:


def preprocess(x, y, mask):
    '''
    preprocess.prepate_data()의 결과를 GCN에 직접 입력으로 넣을 수 있도록 처리해줌
    NUM_EXAMPLES는 window를 적용한 후의 전체 TIME의 갯수이고
    NUM_TIMES가 실제 train에 사용할 TIME의 갯수를 의미함
    NUM_EXAMPLES와 NUM_TIMES가 다른 이유는 전체 NUM_EXAMPLES 중에서 label에 균형을 맞추어
    일부 TIME의 일부 NODE에 대해서만 training을 진행하기 때문임
    
    :param: x (NUM_EXAMPLES, NUM_NODES, WINDOW_SIZE, 8)
    :param: y (NUM_EXAMPLES, NUM_NODES)
    :param: mask (NUM_TIMES, NUM_NODES+1)
    
    :return (NUM_TIMES, x', y', mask')
    x': numpy array of shape (NUM_TIMES, NUM_NODES, WINDOW_SIZE*8) which is flattened over WINDOW_SIZE
    y': numpy array of shape (NUM_TIMES, NUM_NODES, NUM_CLASSES) which is one-hotted
    mask': numpy array of shape (NUM_TIMES, NUM_NODES) which remove first element(TIME) in original mask
    '''
    
    x_ = np.reshape(x, [x.shape[0], x.shape[1], x.shape[2]*x.shape[3]])
    y_ = to_categorical(y) # one-hot encoding
    
    # apply times in mask
    x_ = x_[mask[:, 0], :, :]
    y_ = y_[mask[:, 0], :, :]
    mask_ = mask[:, 1:]
    
    return x_, y_, mask_


# In[ ]:


def print_epoch(epoch, loss_train, true_train, pred_train, loss_valid, true_valid, pred_valid):
    loss_tr = np.mean(loss_train)
    acc_tr = accuracy_score(true_train, pred_train)
    f1_tr = f1_score(true_train, pred_train, average='weighted')
    loss_vd = np.mean(loss_valid)
    acc_vd = accuracy_score(true_valid, pred_valid)
    f1_vd = f1_score(true_valid, pred_valid, average='weighted')
    logging.info('Epoch: %3d, loss_tr: %.4f, acc_tr: %.4f, f1_tr: %.4f, loss_vd: %.4f, acc_vd: %.4f, f1_vd: %.4f'
         % (epoch+1, loss_tr, acc_tr, f1_tr, loss_vd, acc_vd, f1_vd))


# In[ ]:


class GCN():
    '''
    2-layer graph convolutional network
    '''
    
    def __init__(self, input_dim, adj_matrix, gcn_hiddens):
        '''
        GCN constructor
        propagation rule: h' = ADA * h * w
        
        :param: input_dim int length of node's feature
        :param: adj_matrix NxN adjacent matrix with N nodes
        :param: gcn_hiddens list of length 2 which denotes neruons in each layer
        '''

        self.input_dim = input_dim
        N = len(adj_matrix) # number of nodes in graph
        normalized_matrix = normalize_matrix(adj_matrix)
        self.normalized_matrix = tf.constant(normalized_matrix, tf.float32)
        
        num_classes = 4
        init = tf.initializers.he_normal()
        
        # placeholders
        self.x = tf.placeholder(tf.float32, [N, input_dim])
        self.y = tf.placeholder(tf.float32, [N, num_classes])
        self.mask = tf.placeholder(tf.float32, [N])
        
        # first layer
        self.W1 = tf.Variable(init([input_dim, gcn_hiddens[0]]))
        self.L1 = tf.matmul(self.normalized_matrix, self.x) # shape: (N, input_dim)
        self.L1 = tf.matmul(self.L1, self.W1) # shape: (N, gcn_hiddens[0])
        self.L1 = tf.nn.tanh(self.L1)

        # second layer
        self.W2 = tf.Variable(init([gcn_hiddens[0], gcn_hiddens[1]]))
        self.L2 = tf.matmul(self.normalized_matrix, self.L1) # shape: (N, gcn_hiddens[0])
        self.L2 = tf.matmul(self.L2, self.W2) # shape: (N, gcn_hiddens[1])
        self.L2 = tf.nn.relu(self.L2)
        
        # last layer
        self.W3 = tf.Variable(init([gcn_hiddens[1], num_classes]))
        self.L3 = tf.matmul(self.normalized_matrix, self.L2) # shape: (N, gcn_hiddens[1])
        self.L3 = tf.matmul(self.L3, self.W3) # shape: (N, num_classes)
        
        # loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.L3, labels=self.y)
        self.loss = self.loss * self.mask # apply mask
        self.loss = tf.reduce_mean(self.loss)
        
        # prediction
        self.pred = tf.argmax(self.L3, axis=1)
        
    def fit(self, data, save_dir, learning_rate=0.0001, epochs=500, patience=30):
        '''
        train using given dataset and hyper-parameters and then save model to save_dir
        because we use validation set, we will save model to save_dir everywhen there comes best validation loss
        also we use early stopping with default patience as 20
        (e.g. stop training if validation loss does not get better during consecutive 20 epochs)
        
        :param: data tuple of 3 dataset(train, valid, test) which is output of preprocess.prepare_data()
        :param: save_dir path where trained model and train results will be saved
        :param: learning_rate
        :param: epochs
        '''

        logging.info('start fit function')
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(self.loss)
        
        y_true = tf.argmax(self.y, axis=1)
        y_pred = tf.argmax(self.L3, axis=1)
        
        # accuracy and f1 score
        acc = tf.contrib.metrics.accuracy(labels=y_true, predictions=y_pred)
        f1 = tf.contrib.metrics.f1_score(labels=y_true, predictions=y_pred)
        
        # preprocess data for GCN
        # x: (NUM_EXAMPLES, NUM_NODES, WINDOW_SIZE*8)
        # y: (NUM_EXAMPLES, NUM_NODES, NUM_CLASSES)
        # mask: (NUM_EXAMPLES, NUM_NODES)
        train, valid, test = data
        x_train, y_train, mask_train = preprocess(*train)
        x_valid, y_valid, mask_valid = preprocess(*valid)
        x_test, y_test, mask_test = preprocess(*test)
        
        assert x_train.shape[0] == y_train.shape[0] == mask_train.shape[0]
        assert x_valid.shape[0] == y_valid.shape[0] == mask_valid.shape[0]
        assert x_test.shape[0] == y_test.shape[0] == mask_test.shape[0]
        assert self.input_dim == x_train.shape[2]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            min_valid_loss = None
            patience_counter = 0
            
            for epoch in range(epochs):
                # shuffle train set
                p = np.random.permutation(x_train.shape[0])
                x_train_ = x_train[p]
                y_train_ = y_train[p]
                mask_train_ = mask_train[p]
                
                # iter over trainset
                num_iter_train = x_train_.shape[0]
                loss_train = []
                true_train = []
                pred_train = []
                
                for i in range(x_train_.shape[0]):
                    l, true, pred, _ = sess.run([self.loss, y_true, y_pred, train_step], feed_dict={
                        self.x: x_train_[i],
                        self.y: y_train_[i],
                        self.mask: mask_train_[i]
                    })
                    loss_train.append(l)
                    true_train.extend(true[np.argwhere(mask_train_[i]).reshape(-1)].tolist())
                    pred_train.extend(pred[np.argwhere(mask_train_[i]).reshape(-1)].tolist())                    
                    
                # iter over validset
                num_iter_valid = x_valid.shape[0]
                loss_valid = []
                true_valid = []
                pred_valid = []
                
                for i in range(x_valid.shape[0]):
                    l, true, pred = sess.run([self.loss, y_true, y_pred], feed_dict={
                        self.x: x_valid[i],
                        self.y: y_valid[i],
                        self.mask: mask_valid[i]
                    })
                    loss_valid.append(l)
                    true_valid.extend(true[np.argwhere(mask_valid[i]).reshape(-1)].tolist())
                    pred_valid.extend(pred[np.argwhere(mask_valid[i]).reshape(-1)].tolist())
                    
                # print this epoch
                print_epoch(epoch, loss_train, true_train, pred_train, loss_valid, true_valid, pred_valid)
                
                # early stopping check
                loss_valid = np.mean(loss_valid)
                if min_valid_loss is None or min_valid_loss > loss_valid:
                    min_valid_loss = loss_valid
                    saver.save(sess, os.path.join(save_dir, 'model'))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > 20:
                        break
                               
            # evaluation on testset
            num_iter_test = x_test.shape[0]
            loss_test = []
            true_test = []
            pred_test = []
                
            for i in range(num_iter_test):
                l, true, pred = sess.run([self.loss, y_true, y_pred], feed_dict={
                    self.x: x_test[i],
                    self.y: y_test[i],
                    self.mask: mask_test[i]
                })
                loss_test.append(l)
                true_test.extend(true[np.argwhere(mask_test[i]).reshape(-1)].tolist())
                pred_test.extend(pred[np.argwhere(mask_test[i]).reshape(-1)].tolist())
                
            loss_te = np.mean(loss_test)
            acc_te = accuracy_score(true_test, pred_test)
            f1_te = f1_score(true_test, pred_test, average='weighted')
            logging.info('Evaluation, loss_te: %.4f, acc_te: %.4f, f1_te: %.4f' % (loss_te, acc_te, f1_te))
            
            # valid set classification report
            logging.info(classification_report(true_valid, pred_valid))
            # test set classification report
            logging.info(classification_report(true_test, pred_test))


# In[ ]:


def get_modelname(window_size, after, gcn_hiddens):
    '''
    model을 식별할 수 있는 이름을 지정해주는 함수
    '''
    
    return 'gcn_window{}_after{}_dims{}'.format(window_size, after, gcn_hiddens)


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='preprocessed data dir')
    parser.add_argument('--window_size', type=int, required=True, help='number of frame in window')
    parser.add_argument('--after', type=int, required=True, help='number of after window')
    parser.add_argument('--save_dir', type=str, required=True, help='dir to save train result')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--gcn_hiddens', type=int, nargs='+')
    args = parser.parse_args()

    # log와 ckpt가 저장될 directory 생성
    modelname = get_modelname(args.window_size, args.after, args.gcn_hiddens)
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

    input_dim = args.window_size * 8
    adj_matrix = get_adj_matrix()
    model = GCN(input_dim, adj_matrix, args.gcn_hiddens)
    model.fit(data, save_dir=path, learning_rate=args.lr)
    


# In[ ]:




