# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:06:42 2024

@author: Genio
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rc('axes' ,grid=True)
plt.rc('axes', axisbelow=True)
px2inch = 1/plt.rcParams['figure.dpi']

def rand_grad(t=[0,90],m=[1,2],n=1):
    
    t   = np.array(t)*np.pi/180
    m   = np.array(m)

    theta = t[0] + (t[1]-t[0])*np.random.rand(n,1)
    mag   = m[0] + (m[1]-m[0])*np.random.rand(n,1)

    grad_v = mag*np.hstack([np.cos(theta),np.sin(theta)])
    
    return grad_v


def plot_vec(pos,vec,ax,color='b',txt='',k=0.5):
    
    bbox=dict(boxstyle='round,pad=-0.05', fc='w', ec='none')
    
    ax.quiver(pos[0],pos[1],vec[0],vec[1],angles='xy',scale_units='xy',scale=1,color=color)
    
    if txt != '':
        ax.text(pos[0]+vec[0]*k, pos[1]+vec[1]*k, txt, 
                                                      backgroundcolor='w',
                                                      fontsize=10,
                                                      fontweight ='bold',
                                                      ha='center',
                                                      va='center',
                                                      bbox=bbox)
    
    
def plot_multi_vec(vec,ax,color='b',txt=''):
    
    pos  = np.vstack([np.zeros((1,2)),np.cumsum(vec,axis=0)])
    x, y = pos[:,0][:-1], pos[:,1][0:-1]
    u, v = vec[:,0]     , vec[:,1]
    ax.quiver(x, y, u, v, angles='xy',scale_units='xy',scale=1,color=color)
    
    klim = 0.1
    ax.set_xlim( np.min(x)-klim, np.max(x+u)+klim )
    ax.set_ylim( np.min(y)-klim, np.max(y+v)+klim )
    ax.set_aspect(1)
    
    return x,y,u,v

def momentum_gradient_descent(ax,fi=0.25,mu=0.5,w = np.array([0,0])):
  
    # mu = gradient constant
    # fi = momentum constant
    
    v   = np.array([0,0])
    V,W = v,w
    
    for i,grad in enumerate(grad_v): 
        
        v = fi*v + mu*grad
        w = w    + v
    
        V = np.vstack([V,v])
        W = np.vstack([W,w])
        
        # plots
        plot_vec(W[i]+mu*grad, fi*V[i], ax, color='g')
        plot_vec(W[i]        , mu*grad, ax, color='b')
        plot_vec(W[i]        , v      , ax, color='r',txt=f'w{i+1}')
        
     
    ax.set_title(f'Momentum Gradient Descent \n $\eta:${mu}  $\gamma:${fi}  $w_0:${W[0]}')
    
    return W,V


nplots  = 2
height  = 500
fig, ax = plt.subplots(1,nplots,figsize=(nplots*height*px2inch,height*px2inch),constrained_layout=True)


grad_v = rand_grad(n=5,t=[5,85])
grad_v = np.array([[2,0],
                   [1,2],
                   [0,2],
                   [-1,2]])

w0     = np.array([2,3])

# subpltot 1
#############
W0,V0 = momentum_gradient_descent(ax[0],fi=0,mu=1,w=w0)

# subpltot 2
#############
W1,V1 = momentum_gradient_descent(ax[1],fi=0.25,mu=0.5,w=w0)


klim = 0.1
W    = np.vstack([W0,W1])
ax[0].set_xlim( np.min(W[:,0])-klim, np.max(W[:,0])+klim )
ax[0].set_ylim( np.min(W[:,1])-klim, np.max(W[:,1])+klim )
ax[1].set_xlim( np.min(W[:,0])-klim, np.max(W[:,0])+klim )
ax[1].set_ylim( np.min(W[:,1])-klim, np.max(W[:,1])+klim )

    
plt.savefig('imgs/learning_algorithms.png',dpi=150)
plt.show()



