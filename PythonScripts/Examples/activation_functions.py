# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:47:58 2024

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
plt.rc('lines',linewidth=2)
# grid
plt.rc('axes' ,grid=True)

# # pixel in inches
px2inch = 1/plt.rcParams['figure.dpi']



# %%

plt.close('all')


def sigmoid(z):          return 1/(1+np.exp(-z))
def tanh(z):             return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def relu(z):             return np.maximum(0  ,z)
def leaky_relu(z,k=0.1): return np.maximum(k*z,z)

aspect = 0.5


lmax_x = 5
lmax_y = 1*1.1
dxy    = 1
z      = np.linspace(-lmax_x,lmax_x)
fig, ax = plt.subplots(2,2, constrained_layout=True,figsize=(800*px2inch , 500*px2inch))

ax[0][0].set_title('$Sigmoid=\sigma(z)=1/(1+e^{-z})$')
ax[0][0].plot(z,sigmoid(z) );
ax[0][0].set_xlim(-lmax_x*dxy,lmax_x*dxy)
ax[0][0].set_ylim(-lmax_y*dxy,lmax_y*dxy)
ax[0][0].set_box_aspect(aspect)
ax[0][0].set_xlabel('z')

ax[0][1].set_title('$Tanh = (e^z-e^{-z})/(e^z+e^{-z})$')
ax[0][1].plot(z,tanh(z)    );
ax[0][1].set_xlim(-lmax_x*dxy,lmax_x*dxy)
ax[0][1].set_ylim(-lmax_y*dxy,lmax_y*dxy)
ax[0][1].set_box_aspect(aspect)
ax[0][1].set_xlabel('z')

lmax_x = 5
lmax_y = 5
dxy    = 1
z      = np.linspace(-lmax_x,lmax_x,500)

ax[1][0].set_title('$Relu=max(0,z)$')
ax[1][0].plot(z,relu(z) );
ax[1][0].set_xlim(-lmax_x*dxy,lmax_x*dxy)
ax[1][0].set_ylim(-lmax_y*dxy,lmax_y*dxy)
ax[1][0].set_box_aspect(aspect)
ax[1][0].set_xlabel('z')

ax[1][1].set_title('$LeakyRelu=max(k·z,z)$')
ax[1][1].plot(z,leaky_relu(z,0.25) );
ax[1][1].set_xlim(-lmax_x*dxy,lmax_x*dxy)
ax[1][1].set_ylim(-lmax_y*dxy,lmax_y*dxy)
ax[1][1].set_box_aspect(aspect)
ax[1][1].set_xlabel('z')


fig.savefig('imgs/activations_functions.png', dpi=150)

plt.show()