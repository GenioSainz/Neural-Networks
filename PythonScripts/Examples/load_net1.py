# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:56:08 2024

@author: Genio
"""

import matplotlib.pyplot as plt
import numpy as np
import mnist_loader
import network2


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data     = list(test_data)
net           = network2.load('net_save.json')


# %%
  
BCE_cost = lambda a,y: -y*np.log(a)-(1-y)*np.log(1-a)


n        = np.random.randint(len(training_data))
x,y      = training_data[n]

noise    = 0.0
x        = x+noise*np.random.randn(28*28,1)
#x        = np.where(x<0.5,0,1)
a        = net.feedforward(x)
cost     = BCE_cost(a,y)

predict_digit = np.argmax(a)
true_digit    = np.argmax(y)
    


print('y',end='\n')
print(y,end='\n\n')

print('a',end='\n')
print(a,end='\n\n')

print('cost',end='\n')
print(cost,end='\n\n')

size    = 500
nplots  = 3
px2inch = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1,nplots,figsize=(size*nplots*px2inch,size*px2inch),constrained_layout=True)
digits  = np.arange(10)


legend_loc = 0
bar_color  = 'tab:blue'
plot_color = 'tab:red'

ax[0].bar(digits,a.flatten(),color=bar_color,label='Activations')
ax[0].plot(np.nan,np.nan,plot_color,label='Cost',marker='o',mfc='tab:green',mec='k')
ax[0].set_xlabel('Output Neuron')
ax[0].set_ylim(0,1)
ax[0].set_xticks(digits)
ax[0].set_title(f'Digit: Predict:{predict_digit} True:{true_digit}\n TotalCost:{np.sum(cost):0.3f}')
ax[0].legend(loc=legend_loc)
ax_left1 = ax[0].twinx()
ax_left1.plot(digits,cost,plot_color,marker='o',mfc='tab:green',mec='k')
ax[0].tick_params(axis='y', labelcolor=bar_color)
ax_left1.tick_params(axis='y', labelcolor=plot_color)
ax_left1.set_ylim(0,0.2)

ax[1].bar(digits,a.flatten(),log=True,color=bar_color,label='Activations')
ax[1].plot(np.nan,np.nan,plot_color,label='Cost',marker='o',mfc='tab:green',mec='k')
ax[1].set_xlabel('Output Neuron')
ax[1].set_xticks(digits)
ax[1].set_title(f'Digit: Predict:{predict_digit} True:{true_digit}\n TotalCost:{np.sum(cost):0.3f}')
ax[1].legend(loc=legend_loc)
ax_left2 = ax[1].twinx()
ax_left2.semilogy(digits,cost,plot_color,marker='o',mfc='tab:green',mec='k')
ax[1].tick_params(axis='y', labelcolor=bar_color)
ax_left2.tick_params(axis='y', labelcolor=plot_color)
ax_left2.set_ylim(0,0.2)

ax[2].imshow( x.reshape((28,28)) )
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title(f'N train:{n}')
plt.show()
