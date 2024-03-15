# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 00:23:33 2023

@author: Genio Sainz
"""

import numpy as np
import random
import time

def sigmoid_diff(z): return np.exp(-z)/(1+np.exp(-z))**2
def sigmoid(z):      return 1/(1+np.exp(-z))


class BCE(object): # BinaryCrossEntropy

    @staticmethod
    def fn(a, y): return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)
    
class MSE(object): # Mean Square Error
    
    @staticmethod
    def fn(a, y): return 0.5*np.linalg.norm(y-a)**2

    @staticmethod
    def delta(z, a, y): return (a-y) * sigmoid_diff(z)


class Network():
    
    def __init__(self, sizes, X1, X2, cost=BCE):
        
        self.num_layers = len(sizes)
        self.rows       = sizes[1:]
        self.cols       = sizes[:-1]
        self.biases     = [np.random.randn(i,1)            for i   in self.rows]
        self.weights    = [np.random.randn(i,j)/np.sqrt(j) for i,j in zip(self.rows,self.cols)]
        self.cost       = cost
        
        # evaluation 2D domain
        self.X1 = X1.flatten()
        self.X2 = X2.flatten()

    def feedForward(self,a):
        """Return the output of the network if ``a`` is input"""
        
        for w,b in zip(self.weights,self.biases):
            
            a = sigmoid( w@a + b)
            
        return a
    
    def eval2Ddomain(self,N=200):

        Zc    = [] # discrete   values
        Zd    = [] # continuous values
        
        for x1i,x2i in zip(self.X1,self.X2):
            x = np.array([x1i,x2i]).reshape(2,1)
            y = self.feedForward(x)
            Zc.append(np.max(y)   ) 
            Zd.append(np.argmax(y)) 

        Zc = np.array(Zc).reshape(N,N)
        Zd = np.array(Zd).reshape(N,N)

        return [Zc,Zd]
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, eval_2D_train=False, test_data=None):

        print("Neural Network Train Start",end='\n')
        
        start_time     = time.time()
        n              = len(training_data)
        m              = mini_batch_size
         
        # if test_data
        accuracy_array = []
        cost_array     = []
        epoch_array    = []

        # if eval_domain
        animation_data = []

        for epoch in range(epochs+1):
            
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+m] for k in range(0, n, m) ]
            
            for mini_batch in mini_batches:
                """ Gradient descent step per mini batch """
                self.update_mini_batch(mini_batch, eta)
             
            if test_data and epoch%2==0:
               acurracy,cost = self.evaluate_net(test_data)
               accuracy_array.append(acurracy)
               cost_array    .append(cost)
               epoch_array   .append(epoch)

            if eval_2D_train:
               animation_data.append( self.eval2Ddomain() )


        time_end = time.time()-start_time
        print(f"Neural Network Train End: Lapse Time: {time_end:0.3f} s",end='\n')

        if test_data:
            
           train_results = {}
           train_results['accuracy_array'] = accuracy_array
           train_results['cost_array'    ] = cost_array
           train_results['epoch_array'   ] = epoch_array
           train_results['Zcd_end'       ] = self.eval2Ddomain()
       
           return train_results
              
        if eval_2D_train: 
           return animation_data
        
    def update_mini_batch(self, mini_batch, eta):

        nabla_b_sum = [np.zeros(b.shape) for b in self.biases ]
        nabla_w_sum = [np.zeros(w.shape) for w in self.weights]
        m           = len(mini_batch)
        
        """ Sum of Gradients over Mini Bach """
        for x, y in mini_batch:
            """ nabla_bx nabla_wx for single train input"""
            nabla_bx,nabla_wx = self.backprop(x, y)
            
            nabla_b_sum = [nbs+nbx for nbs, nbx in zip(nabla_b_sum, nabla_bx)]
            nabla_w_sum = [nws+nwx for nws, nwx in zip(nabla_w_sum, nabla_wx)]
        
        """ Gradient descent step with mean Gradient over Mini Bach """
        self.biases  = [b-(eta/m)*nbs for b, nbs in zip(self.biases,  nabla_b_sum)] 
        self.weights = [w-(eta/m)*nws for w, nws in zip(self.weights, nabla_w_sum)]
         
        
    def backprop(self, x, y):

        nabla_bx = [np.zeros(b.shape) for b in self.biases ]
        nabla_wx = [np.zeros(w.shape) for w in self.weights]
        
        """forward pass"""
        a      = x
        z_list = [ ] # list to store all the weight sum  z vectors, layer by layer
        a_list = [x] # list to store x and all the activations as a vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            
            z = w@a + b
            a = sigmoid(z)
            
            z_list.append(z)
            a_list.append(a)
            
        """backward pass L""" 
        #delta       = self.cost_diff(a_list[-1], y) * self.sigmoid_diff( z_list[-1] )
        delta        = (self.cost).delta( z_list[-1], a_list[-1], y)
        nabla_bx[-1] = delta
        nabla_wx[-1] = delta @ a_list[-2].T
        
        """backward pass L-1, L-2, L-3 ...
        layers = [a,b,c,d,e] --> len = 5
        [-l for l in range(2, len)] --> [-2,-3,-4]""" 
        for l in range(2, self.num_layers):
            
            delta        = (self.weights[-l+1].T@delta) * sigmoid_diff( z_list[-l] )
            nabla_bx[-l] = delta
            nabla_wx[-l] = delta @ a_list[-l-1].T
    
        return (nabla_bx, nabla_wx)     
    
    def evaluate_net(self, test_data):
        N        = len(test_data)
        acurracy = []
        cost     = []
        for (x, y) in test_data:
            
            aL = self.feedForward(x)
            acurracy.append( np.argmax(aL)==np.argmax(y)  )
            cost    .append( (self.cost).fn(aL,y) )
        
        acurracy = 100*np.sum(acurracy)/N
        cost     = np.sum(cost)/N
    
        return acurracy,cost

   