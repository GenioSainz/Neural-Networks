# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:48:33 2023

@author: Genio
"""
import matplotlib.pyplot as plt
import mnist_loader
import numpy as np
import networkG2 
import network2 

MONITOR = True
epochs  = 30
mini_batch_size = 10
eta    = 0.5
lmbda  = 5
layers = [784, 100, 10]


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


net = network2.Network(layers, cost=network2.CrossEntropyCost)
net.large_weight_initializer()
 
EC,EA,TC,TA = net.SGD(training_data, epochs, mini_batch_size, eta,
                      lmbda                        = lmbda,
                      evaluation_data              = test_data,
                      monitor_evaluation_cost      = MONITOR, 
                      monitor_evaluation_accuracy  = MONITOR,
                      monitor_training_cost        = MONITOR, 
                      monitor_training_accuracy    = MONITOR)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = EC,EA,TC,TA

evaluation_cost     = np.array(evaluation_cost)
evaluation_accuracy = np.array(evaluation_accuracy)

training_cost     = np.array(training_cost)
training_accuracy = np.array(training_accuracy)

#%%

#net.save('net_save.json')

plt.rcParams.update(plt.rcParamsDefault)

SMALL_SIZE  = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

# fonts
plt.rc('font',  size=SMALL_SIZE)
# title
plt.rc('axes',titlesize=BIGGER_SIZE)
plt.rc('axes',titleweight='bold')
# xy-labells
plt.rc('axes',labelsize=SMALL_SIZE)
# xy-ticks
plt.rc('xtick',labelsize=SMALL_SIZE)
plt.rc('ytick',labelsize=SMALL_SIZE)
# legend
plt.rc('legend',fontsize =13)
plt.rc('legend',framealpha=1)
plt.rc('legend',loc='upper center')
# lines
plt.rc('lines',linewidth=1.5)
# grid
plt.rc('axes' ,grid=True)
plt.rc('axes', axisbelow=True)

# pixel in inches
px2inch = 1/plt.rcParams['figure.dpi']
plt.close('all')


len_train = 50000 
len_test  = 10000 

plt.close('all')
px2inch = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax = plt.subplots(1,2, constrained_layout=True,figsize=(1000*px2inch , 500*px2inch))

txt = f'$Layers:{layers}$   $Epochs:{epochs}$\n$m:{mini_batch_size}$   $\eta:{eta}$   $\lambda:{lmbda}$'
fig.suptitle(txt, fontsize=16)

epoch_range = np.arange(1,epochs+1)

ax[0].set_title('COST')
ax[0].plot(epoch_range, training_cost   ,label='Training')
ax[0].plot(epoch_range, evaluation_cost ,label='Test')
ax[0].legend(loc=1)

# abs_error = np.abs(evaluation_cost-training_cost)
# ax_left1  = ax[0].twinx()
# ax_left1.plot(epoch_range,abs_error)

ax[1].set_title('ACCURACY')
ax[1].plot(epoch_range, 100*training_accuracy  /len_train ,label='Training')
ax[1].plot(epoch_range, 100*evaluation_accuracy/len_test  ,label='Test')
ax[1].legend(loc=4)

plt.show()
