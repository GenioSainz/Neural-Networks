# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 00:23:33 2023

@author: Genio
"""

import numpy as np

class Network():

    
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)
        self.rows       = sizes[1:]
        self.cols       = sizes[:-1]
        self.biases     = [np.random.randn(i,1) for i   in self.rows]
        self.weights    = [np.random.randn(i,j) for i,j in zip(self.rows,self.cols)]
    
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    def sigmoid_diff(self,z):
        return np.exp(-z)/(1+np.exp(-z))**2
    
    
    def feedForward(self,a):
        
        for w,b in zip(self.weights,self.biases):
            
            a = self.sigmoid( w@a + b)
            
        return a
    

sizes = [20,3,4,5]
x     = np.random.randn(sizes[0],1)
net0  = Network(sizes)

n = 100
m = 4
x = np.arange(n)

baches = [x[i:i+m] for i in range(0,n,m)]

print(baches)