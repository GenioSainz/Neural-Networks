# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:43:14 2023

@author: Genio
"""

import mnist_loader
import time
start = time.time()

import network
#      50.000           10.000     10.000
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 5,5, 10])
   #SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
net.SGD(      training_data, 30,     10,              3.0, test_data=test_data)


print(time.time() - start)