# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:09:44 2020

@author: vernika
"""
from cnn.lenet import LeNet
from keras.utils import plot_model

# Initialize LeNet and then write the network architecture visualization grpah to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)