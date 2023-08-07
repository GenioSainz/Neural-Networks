# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:26:16 2023

@author: Genio
"""

import numpy as np


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
