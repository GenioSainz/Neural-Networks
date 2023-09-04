# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:43:14 2023

@author: Genio
"""
import mnist_loader

from mainClassNielsen import Network as Network_N
from mainClassGenio   import Network as Network_G

#SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):   
layers          = [784, 30, 30, 10]
epochs          = 1
mini_batch_size = 10
eta             = 1


# training_data = [(x1,y1),(x2,y2)...(xn,yn)]
# xi =  array(784,1)
# yi =  array(10,1)

# test_data = [(x1,y1),(x2,y2)...(xn,yn)]
# xi =  array(784,1)
# yi =  number 0,1,2...9

#      50.000           10.000     10.000
training_dataN, validation_dataN, test_dataN = mnist_loader.load_data_wrapper()
training_dataG, validation_dataG, test_dataG = mnist_loader.load_data_wrapper()


# train and test Nielsen Net
netN = Network_N(layers)
netN.SGD(training_dataN, epochs, mini_batch_size, eta, test_dataN)


# train and test Genio Net
netG = Network_G(layers)
netG.SGD(training_dataG, epochs, mini_batch_size, eta, test_dataG)


