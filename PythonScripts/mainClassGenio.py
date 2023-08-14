# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 00:23:33 2023

@author: Genio
"""

import numpy as np
import random
import time

class Network():

    
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)
        self.rows       = sizes[1:]
        self.cols       = sizes[:-1]
        self.biases     = [np.random.randn(i,1) for i   in self.rows]
        self.weights    = [np.random.randn(i,j) for i,j in zip(self.rows,self.cols)]
    
    
    def sigmoid(self,z):
        """Return sigmoid(z)"""
        
        return 1/(1+np.exp(-z))
    
    
    def sigmoid_diff(self,z):
        """Return sigmoid'(z)"""
        
        return np.exp(-z)/(1+np.exp(-z))**2
 
    
    def feedForward(self,a):
        """Return the output of the network if ``a`` is input"""
        
        for w,b in zip(self.weights,self.biases):
            
            a = self.sigmoid( w@a + b)
            
        return a
    
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        """Train the neural network using mini-batch stochastic
        The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially"""
        print("Genio Neural Network",end='\n')
        
        training_data = list(training_data)
        start_time    = time.time()
        n             = len(training_data)
        m             = mini_batch_size
        
        if test_data:
            test_data = list(test_data)
            n_test    = len(test_data)
        
        for epoch in range(epochs):
            
            random.shuffle(training_data)
            mini_batches = [ training_data[k:k+m] for k in range(0, n, m) ]
            
            for mini_batch in mini_batches:
                """ Gradient descent step per mini batch """
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                eval_true = self.evaluate(test_data)
                print("Epoch {} : {} / {} --> {} %".format(epoch,eval_true,n_test,100*eval_true/n_test))
            else:
                print("Epoch {} complete".format(epoch))
        print("Lapse Time: {} s".format(time.time()-start_time),end='\n \n')
            
        
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m       = len(mini_batch)
        
        """ Sum of Gradients over Mini Bach """
        for x, y in mini_batch:
            
            nabla_bx,nabla_wx = self.backprop(x, y)
            nabla_b = [nb+nbx for nb, nbx in zip(nabla_b, nabla_bx)]
            nabla_w = [nw+nwx for nw, nwx in zip(nabla_w, nabla_wx)]
        
        """ Gradient descent step with mean Gradient over Mini Bach """
        self.weights = [w-(eta/m)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b-(eta/m)*nb for b, nb in zip(self.biases,  nabla_b)] 
         
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_bx, nabla_wx)`` representing the
        gradient for the cost function C_x.  ``nabla_bx`` and
        ``nabla_wx`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_bx = [np.zeros(b.shape) for b in self.biases ]
        nabla_wx = [np.zeros(w.shape) for w in self.weights]
        
        """forward pass"""
        z      = 0
        a      = x
        z_list = [ ] # list to store all the weight sum  z vectors, layer by layer
        a_list = [x] # list to store x and all the activations as a vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            
            z = w@a + b
            a = self.sigmoid(z)
            
            z_list.append(z)
            a_list.append(a)
            
        """backward pass L""" 
        delta        = self.cost_diff(a_list[-1], y) * self.sigmoid_diff( z_list[-1] )
        nabla_bx[-1] = delta
        nabla_wx[-1] = delta @ a_list[-2].T
        
        """backward pass L-1, L-2, L-3 ...
        layers = [a,b,c,d,e] --> len = 5
        [-l for l in range(2, len)] --> [-2,-3,-4].""" 
        for l in range(2, self.num_layers):
            
            delta        = (self.weights[-l+1].T@delta) * self.sigmoid_diff(z_list[-l])
            nabla_bx[-l] = delta
            nabla_wx[-l] = delta @ a_list[-l-1].T
    
        return (nabla_bx, nabla_wx)     
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        test_results = [ (np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    
    def cost_diff(self, aL, y):
        """Return the vector of partial derivatives partial(C_x)/partial(aL)
        for the output activations MSE C=||aL-y||^2"""
        
        return 2*(aL-y)


     
        