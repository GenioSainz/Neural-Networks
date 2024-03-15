# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:03:37 2024

@author: Genio
"""

import numpy as np
import matplotlib.pyplot as plt
import time

class Network():

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.rows       = sizes[1:]
        self.cols       = sizes[:-1]
        self.biases     = [np.random.randn(i, 1) for i    in self.rows]
        self.weights    = [np.random.randn(i, j) for i, j in zip(self.rows, self.cols)]
    
    def cost_diff(self, aL, y):
        return (aL-y)
  
    def sigmoid(self, z): 
        return 1/(1+np.exp(-z))
    
    def sigmoid_diff(self,z):
        return np.exp(-z)/(1+np.exp(-z))**2

    def feedForward(self, a):
        
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(w@a + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data,evaluate=False):
        
        print("Genio Neural Network",end='\n')
        start_time    = time.time()
        
        training_data = list(training_data)
        test_data     = list(test_data)
        n             = len(training_data)
        m             = mini_batch_size
        
        trainC = []
        testC  = []
        for epoch in range(epochs):
            
            np.random.shuffle(training_data)
            mini_batches = [ training_data[k:k+m] for k in range(0, n, m) ]
            
            for mini_batch in mini_batches:
                """ Gradient descent step per mini batch """
                self.update_mini_batch(mini_batch, eta)
            if evaluate:  
               train_cost,test_cost = self.evaluate( training_data, test_data)
               trainC.append(train_cost)
               testC.append(test_cost)
               # print(f"Epoch:{epoch}  TrainCost:{train_cost:0.3f}  TestCost:{test_cost:0.3f}")
                
        print("Lapse Time: {} s".format(time.time()-start_time),end='\n \n')
            
        return np.array(trainC), np.array(testC)
    
    
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
            a = self.sigmoid(z)
            
            z_list.append(z)
            a_list.append(a)
            
        """backward pass L""" 
        delta        = self.cost_diff(a_list[-1], y) * self.sigmoid_diff( z_list[-1] )
        nabla_bx[-1] = delta
        nabla_wx[-1] = delta @ a_list[-2].T
        
        """backward pass L-1, L-2, L-3 ..."""
        for l in range(2, self.num_layers):
            
            delta        = (self.weights[-l+1].T@delta) * self.sigmoid_diff( z_list[-l] )
            nabla_bx[-l] = delta
            nabla_wx[-l] = delta @ a_list[-l-1].T
    
        return (nabla_bx, nabla_wx)  
    
    
    def evaluate(self, training_data, test_data):
        
        train_cost = sum( [(y[0][0]-self.feedForward(x)[0][0])**2 for x,y in training_data] )/len(training_data)
        test_cost  = sum( [(y[0][0]-self.feedForward(x)[0][0])**2 for x,y in test_data    ] )/len(test_data)
        
        return train_cost,test_cost
   
    
   
f = lambda x: 0.2 + 0.4*x**2 + 0.3*x*np.sin(15*x) + 0.05*np.cos(50*x)

def get_xy_data(f,x):
    
    y  = f(x)
    xx = [np.array([[xi]]) for xi in x]
    yy = [np.array([[yi]]) for yi in y]
    
    return zip(xx,yy)

n      = 500
xtrain = np.linspace(0,1,n)
xtest  = np.random.rand(n)    

training_data   = get_xy_data(f,xtrain)
test_data       = get_xy_data(f,xtest)

layers          = [1,100,1]
epochs          = 800
mini_batch_size = 1
eta             = 3

net           = Network(layers)
trainC, testC = net.SGD(training_data, epochs, mini_batch_size, eta,test_data,evaluate=True)


# %%

px2inch = 1/plt.rcParams['figure.dpi']
plt.close('all')

fig, ax = plt.subplots(1,2, constrained_layout=True,figsize=(800*px2inch, 500*px2inch))


txt = f'$Layers:{layers}$   $Epochs:{epochs}$\n$m:{mini_batch_size}$   $\eta:{eta}$ '

fig.suptitle(txt, fontsize=16)
ax[0].plot( np.arange(epochs), trainC, label='trainC')
ax[0].plot( np.arange(epochs), testC,  label='testC')
ax[0].legend()

n = 250
x = np.linspace(0,1,n) 
y = f(x)

ax[1].plot(x,y)


xr = np.sort( np.random.rand(n) )
yr = [net.feedForward( np.array([[xi]]) )[0][0] for xi in x]

ax[1].plot(xr,yr)


plt.show()