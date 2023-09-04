# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:06:34 2023

@author: Genio
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:22:07 2023

@author: Genio
"""
import numpy as np
import random
import time
import matplotlib.pyplot as plt

import mnist_loader

start = time.time()



#      50.000           10.000     10.000 
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# training_data          [ ( x           , y),...         ]
# list(training_data) => [ (array(784,1),array(10,1)), ...]

# validation_data          [ ( x          , y ), ...]
# list(validation_data) => [ (array(784,1),int), ...]

# test_data          [ ( x          , y ), ...]
# list(test_data) => [ (array(784,1),int), ...]

lapse = time.time() - start
print(f'MNIST loader: {lapse:03} seconds')


x_train = list(training_data)
random.shuffle( x_train )

sum_digits = [np.zeros((28,28))  for i in range(10)]
sum_count  = np.zeros((1,10)).flatten()


## compute means and counts digits
#####################################
for i in range(len(x_train)):
    
    x,y  = x_train[i]
    x    = np.reshape(x,(28,28))
    y    = y.flatten()
    indx = np.where(y == 1)[0][0]
    
    
    sum_digits[indx] = sum_digits[indx] + x
    sum_count[indx]  = sum_count[indx]  + 1
    
lapse = time.time() - start
print(f'x_train: {lapse:03} seconds')


## plot mean digits
##############################
plt.close('all')
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, axs = plt.subplots(2,5, constrained_layout=True,figsize=(1800*px, 900*px))

count = 0
for ax in axs.flat:
    
    x = sum_digits[count]/sum_count[count]
    
    ax.set_title(f'N: {count}',fontsize=8)
    ax.imshow(x)
    ax.set_xticks([])
    ax.set_yticks([])
        
    count=count+1
    

## plot mean digits reshape to line
#####################################
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig2, axs2 = plt.subplots(2,5, constrained_layout=True,figsize=(1800*px, 900*px))

count = 0
for ax2 in axs2.flat:
    
    x = sum_digits[count]/sum_count[count]
    x = np.reshape(x,(1,28*28)).flatten()
    
    ax2.plot(x,'b')
    
    meanM = [300]
    for window in meanM:
        
        meanX = np.convolve(x, np.ones(window)/window, mode='same')
    
        ax2.plot(meanX,'r')
        ax2.grid(True)
    
    ax2.set_ylim(0,1) 
    
    count=count+1

        
plt.show()