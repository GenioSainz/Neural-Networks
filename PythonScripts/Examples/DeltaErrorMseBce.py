# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:30:59 2023

@author: Genio
"""

import numpy as np
import matplotlib.pyplot as plt

# set defaults
plt.rcParams.update(plt.rcParamsDefault)

SMALL_SIZE  = 12
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
plt.rc('legend',fontsize =SMALL_SIZE)
plt.rc('legend',framealpha=1)
plt.rc('legend',loc='upper center')
# lines
plt.rc('lines',linewidth=1.5)
# grid
plt.rc('axes' ,grid=True)

# pixel in inches
px2inch = 1/plt.rcParams['figure.dpi']

def sigmoid(z):          return 1/(1+np.exp(-z))
def sigmoid_diff(z):     return sigmoid(z)*(1-sigmoid(z))

def quadratic_cost(a,y): return  ( y-a )**2
def cross_entropy(a,y):  return -( y*np.log(a) + (1-y)*np.log(1-a) )

def delta_mse(z,a,y):    return (a-y)*sigmoid_diff(z)
def delta_bce(z,a,y):    return (a-y)

plt.close('all')
fig, ax = plt.subplots(1,2, constrained_layout=True,figsize=(1000*px2inch , 450*px2inch))

z     = np.linspace(-4,4)
a     = sigmoid(z)

y     = 1
mse   = quadratic_cost(sigmoid(z),y)
bce   = cross_entropy( sigmoid(z),y)
d_mse = delta_mse(z,a,y)
d_bce = delta_bce(z,a,y)
ax[0].plot(z,sigmoid(z),label='$a=\sigma(z)$');ax[0].plot(z,mse,label='$MSE _ cost$');ax[0].plot(z,bce,label='$BCE _ cost$')
ax[0].plot(z,d_mse,label='$\delta^L MSE $');   ax[0].plot(z,d_bce,label='$\delta^L _ BCE$');
ax[0].set_title(f'Desire y = {y}');            ax[0].set_xlabel('z');ax[0].legend();
ax[0].set_ylim(-1,4)

y     = 0
mse   = quadratic_cost(sigmoid(z),y)
bce   = cross_entropy( sigmoid(z),y)
d_mse = delta_mse(z,a,y)
d_bce = delta_bce(z,a,y)
ax[1].plot(z,sigmoid(z),label='$a=\sigma(z)$'); ax[1].plot(z,mse,label='$MSE _ cost$');ax[1].plot(z,bce,label='$BCE _ cost$')
ax[1].plot(z,d_mse,label='$\delta^L MSE $');    ax[1].plot(z,d_bce,label='$\delta^L _ BCE$');
ax[1].set_title(f'Desire y = {y}');             ax[1].set_xlabel('z');ax[1].legend();
ax[1].set_ylim(-1,4)

fig.savefig('imgs/DeltaError.png', dpi=200)

plt.show()