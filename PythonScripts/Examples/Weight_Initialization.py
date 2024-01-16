# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:28:31 2024

@author: Genio
"""

import matplotlib.pyplot as plt
import numpy as np

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
plt.rc('legend',fontsize =SMALL_SIZE)
plt.rc('legend',framealpha=1)
plt.rc('legend',loc='upper center')
# lines
plt.rc('lines',linewidth=1.5)
# grid
plt.rc('axes' ,grid=True)
plt.rc('axes', axisbelow=True)

# pixel in inches
px2inch = 1/plt.rcParams['figure.dpi']

  
n       = 20
layers  = [n,n,n,n]
rows    = layers[1:]
cols    = layers[:-1]
biases  = [np.random.randn(r,1) for r in rows]

weights_mode = 'randn_normalize'
weights_mode = 'randn'

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
fig, ax = plt.subplots(len(Z),2,figsize=(1200*px2inch,800*px2inch),constrained_layout=True)

plt.suptitle(f'Z(weight inputs) / A(activations)\n{n} Neurons/Layers')

for i,(z,a) in enumerate(zip(Z,A)):
    
    hz = z.flatten()
    ha = a.flatten()
    x  = np.arange(len(ha))
    
    ax[i][0].bar(x,hz,label=f'Layer:{i}')
    ax[i][1].bar(x,ha,color='tab:green',label=f'Layer:{i}')
    ax[i][1].set_ylim(-5,5)
    
    ax[i][0].legend(loc=1)
    ax[i][1].legend(loc=1)

plt.show()


# fig.savefig(f'imgs/weights_{weights_mode}.png', dpi=150)

n = 100
x = np.random.randn(n,1)
y = np.random.randn(n,1)/np.sqrt(n)

print(np.std(x)),print(np.var(x))
print(np.std(y)),print(np.var(y))


def get_gauss_curve(u=0,s=1,ns=4):

    x_pdf = np.linspace(u-ns*s,u+ns*s,250)
    y_pdf = 1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_pdf-u)/s)**2)

    return x_pdf,y_pdf

fig2, ax2 = plt.subplots(1,1,constrained_layout=True,figsize=((800*px2inch,600*px2inch)));
u = 0
t = ['weights randn/sqrt(n)\n',
     'weights randn\n']

                                    #[1.22,np.sqrt(51)]
for i,(s ,ns, ti) in enumerate( zip([2,81],[12,3],t) ):
    

    x_pdf,y_pdf = get_gauss_curve(u,np.sqrt(s),ns)
    ax2.plot(x_pdf,y_pdf,label=f'{ti} $\sigma^2$ :{s:6.2f}',zorder=i+5);
    
ax2.legend(loc='upper right')
ax2.set_xlabel('$z = \sum w_j x_j + b$')
ax2.set_title('Gaussian Random WeightSum Z Distribution' )

fig2.savefig('imgs/weight_distribution.png', dpi=150)

plt.show()