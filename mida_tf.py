
# coding: utf-8

import numpy as np
import tensorflow as tf
from utils import _permutation
from sklearn.impute import SimpleImputer



class MidaImputer():
    
    def __init__(self, dim_X, theta, dr):
        
        self.dim_X = dim_X
        self.theta = theta
        self.dr = dr
        
        tf.reset_default_graph()
        self.G = tf.Graph()
        self.G.as_default()
        
        self.X = tf.placeholder(tf.float32, shape=(None, self.dim_X))
        self.trn_flg = tf.placeholder(tf.bool)
        self.pred = self.forward(self.X, self.trn_flg)
        
        
        self.sess = tf.Session()
    def _encoder(self, X, trn_flg):
        
        with tf.variable_scope('encoder', reuse=False):
            X = tf.layers.dropout(X, self.dr, training=trn_flg)
            X = tf.layers.dense(X, self.dim_X + self.theta*1, activation = tf.nn.tanh)
            X = tf.layers.dense(X, self.dim_X + self.theta*2, activation = tf.nn.tanh)
            o = tf.layers.dense(X, self.dim_X + self.theta*3, activation = tf.nn.tanh)
        
        return o
    
    def _decoder(self, X):
        
        with tf.variable_scope('decoder', reuse=False):
            X = tf.layers.dense(X, self.dim_X + self.theta*2, activation = tf.nn.tanh)
            X = tf.layers.dense(X, self.dim_X + self.theta*1, activation = tf.nn.tanh)
            o = tf.layers.dense(X, self.dim_X + self.theta*0, activation = tf.nn.tanh)
            
        return o
    
    def forward(self, X, trn_flg):
        z = self._encoder(X, trn_flg)
        o = self._decoder(z)
        
        return o
    
    def fit(self, X_trn, num_epochs, batch_size, optimizer):
        self.mean_imputer = SimpleImputer()
        X_init = self.mean_imputer.fit_transform(X_trn)
        
        n_batch = int(len(X_trn)/ batch_size)
        
        cost = tf.losses.mean_squared_error(self.pred, self.X)
        
        if optimizer == 'nesterov':
            train_op = tf.train.MomentumOptimizer(learning_rate =0.01, momentum=0.99, use_nesterov=True).minimize(cost)
        elif optimizer == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(cost)
        
        
        self.sess.run(tf.global_variables_initializer())
        
        for epoch in range(num_epochs):
            [X_init] = _permutation([X_init])
            
            for i in range(n_batch):
                start_ = i*batch_size
                end_ = start_ + batch_size
                assert end_ - start_ == batch_size
                
                _, cost_val = self.sess.run([train_op, cost], feed_dict={self.X: X_init[start_:end_], self.trn_flg: True})
            
            if epoch % 100 ==0:
                print('Epoch: %d, Loss: %f'%(epoch+1, cost_val))
            
            if cost_val < 1e-06: break
                
        print('Learning finished !')
    
    def reconstruction(self, X):
        out = self.sess.run(self.pred, feed_dict={self.X: X})
        
        return out
    
    def transform(self, X):
        missing_mask = np.isnan(X).astype(int)
        X_input = self.mean_imputer.transform(X)
        X_reconstructed = self.sess.run(self.pred, feed_dict={self.X: X_input, self.trn_flg: True})
        X_imputed = np.where(missing_mask == 1, X_reconstructed, X)
        
        return X_imputed

