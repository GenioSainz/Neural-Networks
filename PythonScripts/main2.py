# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:48:33 2023

@author: Genio
"""

import mnist_loader
import networkG2 
import network2 


mini_batch_size = 100
epochs = 3
eta = 0.5


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = networkG2.Network([784, 30, 10], cost=networkG2.CrossEntropyCost)
net.large_weight_initializer()
 
net.SGD(training_data, epochs, mini_batch_size, eta,
                    lmbda = 0.1,
                    evaluation_data=test_data,
                    monitor_evaluation_cost=True, 
                    monitor_evaluation_accuracy=True,
                    monitor_training_cost=True, 
                    monitor_training_accuracy=True)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
 
net.SGD(training_data, epochs, mini_batch_size, eta,
                    lmbda = 0.1,
                    evaluation_data=test_data,
                    monitor_evaluation_cost=True, 
                    monitor_evaluation_accuracy=True,
                    monitor_training_cost=True, 
                    monitor_training_accuracy=True)