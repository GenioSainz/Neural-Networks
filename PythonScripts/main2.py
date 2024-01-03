# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:48:33 2023

@author: Genio
"""

import mnist_loader
import network2 


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

#%%

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

net.large_weight_initializer()

net.SGD(training_data[:1000], 400, 10, 0.5,
                            lmbda = 0.1,
                            evaluation_data=test_data,
                            monitor_evaluation_cost=True, 
                            monitor_evaluation_accuracy=True,
                            monitor_training_cost=True, 
                            monitor_training_accuracy=True)

