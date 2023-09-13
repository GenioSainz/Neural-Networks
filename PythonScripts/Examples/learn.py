# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:26:16 2023

@author: Genio
"""

import numpy as np
import random


def indexing():
    
    n = 11
    l = [ i*10 for i in range(n)]
    print(' ')
    print('l       => ',l)
    print('l[0]    => ',l[0])
    print('l[-1]   => ',l[-1])
    print('l[1:]   => ',l[1:]) 
    print('l[:-1]  => ',l[:-1]) 
    
    print(' '),print('Taking n first elements of a list')
    print('l[0:-1] => ',l[0:-1])         
    print('l[:-1]  => ',l[:-1]) 
    print('l[2:5]  => ',l[2:5])
    print('l[0:5]  => ',l[0:5])
    print('l[:5]   => ',l[:5])       
    print('l[1:5]  => ',l[1:5]) 
    
    print(' '),print('Taking n last elements of a list')
    print('l[-3:]   => ',l[-3:])
    print('l[-3:-1] => ',l[-3:-1]) 

    print(' '),print('Taking all but n last elements of a list')   
    print('l[:-2]   => ',l[:-2]);
    print('l[:-3]   => ',l[:-3])  
    
def concat():       
    rows = [2,3]
    cols = [4,5]
    M    = [ np.random.randint(1,9,(i,j))         for i,j, in zip(rows,cols)]
    Mr   = [ np.r_[ np.random.randint(1,9,(i,j)), 100*np.ones((1,j)) ] for i,j, in zip(rows,cols)]
    Mc   = [ np.c_[ np.random.randint(1,9,(i,j)), 200*np.ones((i,1)) ] for i,j, in zip(rows,cols)]
    
    for i in range(len(M)):
        print(' '),print(M[i])
        
    for i in range(len(Mr)):
        print(' '),print(Mr[i])
        
    for i in range(len(Mc)):
        print(' '),print(Mc[i])
        
   # M,Mr,Mc = examples()
    
    return M,Mr,Mc


N = 16
a = np.arange(N).reshape((N,1))
n = int(np.sqrt(N))
b = a.reshape((n,n))

print(b[b > 2]) 


def get_price(price):
    return price if price > 0 else 0

original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
prices          = [i if i > 0 else 0 for i in original_prices]
print(prices)


N         = 20
m         = 4
train_set = np.arange(N)
mini_bach = [train_set[i:i+m] for i in range(0,len(train_set),m)]

print('mini_bach:')
print(train_set)
print(mini_bach)


## without using list ,zip iterator disappears
#################################################
letters = ['a','b','c']
numbers = [1,2,3]
z       = zip(numbers ,letters)

for l,n in z: print('A1',l,n)
for l,n in z: print('A2',l,n)


letters = ['a','b','b','d','e']
numbers = [1,2,3,4,5]
z       = list(zip(numbers ,letters))

for l,n in z: print('B1',l,n)
for l,n in z: print('B2',l,n)


## soft max
##############
## http://neuralnetworksanddeeplearning.com/chap3.html#softmax
print('\n')
print('SOFT MAX')
print('######################')
zL= np.array([2.5,-1,3.2,0.5])
aL = np.exp(zL)/np.sum( np.exp(zL) )
print(np.round(aL,3),'\n')


print('ZIP ENUMERATE')
print('######################')
def sig(z): return 1/(1+np.exp(-z))

net    = [2,3,4,5]
rows   = net[1:]
cols   = net[:-1]
a      = np.ones((net[0],1))
w_list = [np.random.randn(i,j) for i,j in zip(rows,cols)]
b_list = [np.random.randn(i,1) for i   in rows]


for l,(wl,bl) in enumerate(zip(w_list,b_list)):
    
    a = sig(wl@a +bl)
    
    print('layer: ',l)
    print('########')
    print('Wl: ',wl,'\n')
    print('bl: ',bl,'\n')
    print('al: ',a,'\n')
    


def An(n):
     
    # https://rowannicholls.github.io/python/graphs/image_size.html
    
    a = 1.189*0.5**(0.5*n)
    b = 0.841*0.5**(0.5*n)
    
    return a,b

# FORMAT 
X = np.arange(2,13)

for x in X:
    
    txt  = "Number:{:3}  scuare:{:4} number+pi:{:.2f} cube:{:5}".format(x,x**2,x+np.pi,x**3)
    print(txt)