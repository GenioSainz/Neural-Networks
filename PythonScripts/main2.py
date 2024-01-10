# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:48:33 2023

@author: Genio
"""

import mnist_loader
import networkG2 
import network2 

MONITOR = True
epochs = 3
mini_batch_size = 10
eta = 0.5
lmbda = 5

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = networkG2.Network([784, 30, 10], cost=networkG2.CrossEntropyCost)
net.large_weight_initializer()
 
EC,EA,TC,TA = net.SGD(training_data, epochs, mini_batch_size, eta,
                      lmbda                        = lmbda,
                      evaluation_data              = validation_data,
                      monitor_evaluation_cost      = MONITOR, 
                      monitor_evaluation_accuracy  = MONITOR,
                      monitor_training_cost        = MONITOR, 
                      monitor_training_accuracy    = MONITOR)

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = EC,EA,TC,TA
