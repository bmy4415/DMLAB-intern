#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


# In[ ]:


def get_adj_matrix():
    """
    Real map of 301 building
    """
    arr = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]
        
    return arr

def normalize_matrix(adj_matrix):
    '''
    return numpy array which represents normalized matrix, D^(-1/2) A D^(-1/2)
    '''
    ahat = np.array(adj_matrix) + np.eye(len(adj_matrix))
    dhat_prime = np.diag(1 / np.sqrt(np.sum(ahat, axis=1)))
    return np.matmul(np.matmul(dhat_prime, ahat), dhat_prime)

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
        self.L3 = tf.matmul(self.L2, self.W3) # shape: (N, num_classes)
        
        # loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.L3, labels=self.y)
        self.loss = self.loss * self.mask # apply mask
        self.loss = tf.reduce_mean(self.loss)
        
        # prediction
        self.pred = tf.argmax(self.L3, axis=1)


# In[ ]:


def predict_future_people(X):
    '''
    :param: numpy array of shape: (NUM_NODES=4, input_dim=WINDOW_SIZE*8)
    :return: numpy array of shape: (NUM_NODES,) which represents label of each nodes
    '''
    
    result = sess.run(model.pred, feed_dict={
        model.x: X,
    })
    
    print('result:', result)
    
    return result


# In[ ]:


# load model

adj_matrix = get_adj_matrix()
window_size = 8
input_dim = window_size * 8
gcn_hiddens = [64, 64]

model = GCN(input_dim, adj_matrix, gcn_hiddens)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'model')


# In[ ]:




