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
import mnist_loader
import time
import matplotlib.pyplot as plt

start = time.time()

#      50.000           10.000     10.000 
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

lapse = time.time() - start
print(f'MNIST loader: {lapse:03} seconds')


x_train = list(training_data)
random.shuffle( x_train )

sum_digits  = [np.zeros((28,28))  for i in range(10)]
sum_count   = np.zeros((1,10)).flatten()
mean_digits = []

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
    mean_digits.append(x)
    
    ax.set_title(f'N: {count}',fontsize=8)
    ax.imshow(x)
    ax.set_xticks([])
    ax.set_yticks([])
        
    count=count+1
    

## plot mean digits and abs
#####################################
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig2, axs2 = plt.subplots(4,9, constrained_layout=True,figsize=(1800*px, 900*px))

count = 0
digit = 5

for i in range(len(axs2[0])):
    
    ax1 = axs2[0][i]
    ax2 = axs2[1][i]
    ax3 = axs2[2][i]
    ax4 = axs2[3][i]

    
    while True:
          x,y  = x_train[count]
          x    = np.reshape(x,(28,28))
          y    = y.flatten()
          indx = np.where(y == 1)[0][0]
          count = count+1
          if indx == digit:
             break
          
          
    ax1.imshow(x);ax1.set_xticks([]);ax1.set_yticks([])
    
    ax2.imshow(mean_digits[digit]);ax2.set_xticks([]);ax2.set_yticks([])
    
    xx = np.abs( x- mean_digits[indx] );ax3.set_xticks([]);ax3.set_yticks([])
    ax3.imshow(xx)
    
    xx = ( x- mean_digits[indx] )**2
    ax4.imshow(xx);ax4.set_xticks([]);ax4.set_yticks([])
    
    
plt.show()

