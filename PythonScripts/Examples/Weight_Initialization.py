# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:28:31 2024

@author: Genio
"""

import matplotlib.pyplot as plt
import numpy as np
  
n       = 80
layers  = [n,n,n,n]
rows    = layers[1:]
cols    = layers[:-1]
biases  = [np.random.randn(r,1) for r   in rows          ]

weights_mode = 'randn'
weights_mode = 'randn_normalize'

if weights_mode == 'randn':
   weights = [np.random.randn(r,c) for r,c in zip(rows,cols)]
     
elif weights_mode == 'randn_normalize':
     weights = [np.random.randn(r,c)/np.sqrt(c) for r,c in zip(rows,cols)]

a = np.ones( (cols[0],1) )

sigmoid = lambda z: 1/(1+np.exp(-z))
tanh    = lambda z: (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))


def feedForward(a,weights,biases):
    
    Z,A = [],[]
    for w,b in zip(weights,biases):
        
        z = w@a + b
        a = sigmoid(z)
        Z.append(z)
        A.append(a)
        
    return Z,A

Z,A = feedForward(a,weights,biases)


plt.close('all')
px2inch = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(len(Z),2,figsize=(1200*px2inch,800*px2inch),constrained_layout=True)

plt.suptitle(f'Z(weight inputs) / A(activations)\n{n} Neurons/Layers')

for i,(z,a) in enumerate(zip(Z,A)):
    
    hz = z.flatten()
    ha = a.flatten()
    x = np.arange(len(ha))
    
    ax[i][0].bar(x,hz,label=f'Layer:{i}')
    ax[i][1].bar(x,ha,color='tab:green',label=f'Layer:{i}')
    
    ax[i][0].legend(loc=1)
    ax[i][1].legend(loc=1)

plt.show()


#fig.savefig(f'weights_{weights_mode}.png', dpi=150)

n = 100
x = np.random.randn(n,1)
y = np.random.randn(n,1)/np.sqrt(n)

print(np.std(x)),print(np.var(x))
print(np.std(y)),print(np.var(y))