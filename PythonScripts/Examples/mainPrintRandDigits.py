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


cmap_list = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone','pink', 'spring', 'summer', 'autumn', 'winter', 'cool','Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']

def printDigits(nRows,nCols,title=False):
    
    plt.close('all');
    
    random.shuffle( x_train )
    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(nRows, nCols, constrained_layout=True,figsize=(1600*px, 900*px))
    
    count = 0;
    for ax in axs.flat:
        
        x,y  = x_train[count]
        x    = np.reshape(x,(28,28))
        y    = y.flatten()
        indx = np.nonzero(y == 1)
        rani = random.randint(0,len(cmap_list)-1)
        
        if count%10 == 0:
            x = np.random.randn(28,28)
        
        if title: 
            ax.set_title(f'N: {indx[0][0]}',fontsize=8)
            
       
        ax.imshow(x, cmap = cmap_list[rani] )
        ax.set_xticks([])
        ax.set_yticks([])
            
        count=count+1
        
    plt.show()
    
    return axs
        
    
ax = printDigits(9,16)



