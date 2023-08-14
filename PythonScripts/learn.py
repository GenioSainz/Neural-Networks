# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:26:16 2023

@author: Genio
"""

import numpy as np
import random


def indexing():
    
    n = 11
    l = [ i*10 for i in range(n)]
    print(' ')
    print('l       => ',l)
    print('l[0]    => ',l[0])
    print('l[-1]   => ',l[-1])
    print('l[1:]   => ',l[1:]) 
    print('l[:-1]  => ',l[:-1]) 
    
    print(' '),print('Taking n first elements of a list')
    print('l[0:-1] => ',l[0:-1])         
    print('l[:-1]  => ',l[:-1]) 
    print('l[2:5]  => ',l[2:5])
    print('l[0:5]  => ',l[0:5])
    print('l[:5]   => ',l[:5])       
    print('l[1:5]  => ',l[1:5]) 
    
    print(' '),print('Taking n last elements of a list')
    print('l[-3:]   => ',l[-3:])
    print('l[-3:-1] => ',l[-3:-1]) 

    print(' '),print('Taking all but n last elements of a list')   
    print('l[:-2]   => ',l[:-2]);
    print('l[:-3]   => ',l[:-3])  
    
def concat():       
    rows = [2,3]
    cols = [4,5]
    M    = [ np.random.randint(1,9,(i,j))         for i,j, in zip(rows,cols)]
    Mr   = [ np.r_[ np.random.randint(1,9,(i,j)), 100*np.ones((1,j)) ] for i,j, in zip(rows,cols)]
    Mc   = [ np.c_[ np.random.randint(1,9,(i,j)), 200*np.ones((i,1)) ] for i,j, in zip(rows,cols)]
    
    for i in range(len(M)):
        print(' '),print(M[i])
        
    for i in range(len(Mr)):
        print(' '),print(Mr[i])
        
    for i in range(len(Mc)):
        print(' '),print(Mc[i])
        
   # M,Mr,Mc = examples()
    
    return M,Mr,Mc


N = 16
a = np.arange(N).reshape((N,1))
n = int(np.sqrt(N))
b = a.reshape((n,n))

print(b[b > 2]) 


def get_price(price):
    return price if price > 0 else 0

original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
prices          = [i if i > 0 else 0 for i in original_prices]
print(prices)


N         = 20
m         = 4
train_set = np.arange(N)
mini_bach = [train_set[i:i+m] for i in range(0,len(train_set),m)]

print('mini_bach:')
print(train_set)
print(mini_bach)




## without using list ,zip iterator disappears
#################################################
letters = ['a','b','c']
numbers = [1,2,3]
z       = zip(numbers ,letters)

for l,n in z: print('A1',l,n)
for l,n in z: print('A2',l,n)


letters = ['a','b','b','d','e']
numbers = [1,2,3,4,5]
z       = list(zip(numbers ,letters))

for l,n in z: print('B1',l,n)
for l,n in z: print('B2',l,n)



class Net():

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
    
    def SGD(self, training_data, epochs, mini_batch_size, eta):

        training_data = list(training_data)
        n             = len(training_data)
        m             = mini_batch_size

        for epoch in range(epochs):
            
            random.shuffle(training_data)
            mini_batches  = [training_data[i:i+m] for i in range(0,n,m)]
        
            for mini_batch in mini_batches:
                self.update_mini_bach(mini_batch,eta)
                
            print(epoch)

    def update_mini_bach(self,mini_batch,eta):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases ]
        m       = len(mini_batch)
        
        for x,y in mini_batch:
            
            nabla_bx,nabla_wx = self.backprop(x,y)
            nabla_w = [ nw + nwx  for nw,nwx in zip(nabla_w, nabla_wx) ]
            nabla_b = [ nb + nbx  for nb,nbx in zip(nabla_b, nabla_bx) ]
            
        self.weights = [ w-(eta/m)*nw for w,nw in zip(self.weights,nabla_w) ]
        self.biases  = [ b-(eta/m)*nb for b,nb in zip(self.biases ,nabla_b) ]

    def backprop(self, x,y):
    
        nabla_bx = [np.zeros(b.shape) for b in self.biases ]
        nabla_wx = [np.zeros(w.shape) for w in self.weights]
        z      = 0
        a      = x
        z_list = [ ]
        a_list = [x]
        
        """ forward pass """
        for w,b in zip(self.weights,self.biases):
             
            z = w@a + b
            a = self.sigmoid(z)

            z_list.append(z)
            a_list.append(a)

        """ backward pass L""" 
        delta        = self.cost_diff(a_list[-1],y)*self.sigmoid_diff(z_list[-1])
        nabla_wx[-1] = delta@a_list[-2].T
        nabla_bx[-1] = delta
        
        
        for l in range(2,self.num_layers):
    
            delta = (self.weights[-l+1].T@delta)*self.sigmoid_diff(z_list[-l])
            nabla_wx[-l] = delta@a_list[-l-1].T
            nabla_bx[-l] = delta

        return (nabla_bx,nabla_wx)
    
       
    def cost_diff(self,aL,y):
        
        return 2*(aL-y)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        test_results = [ (np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

