"""
  In this file, we :
  - Download the mnist database if not downloaded already
  - Train the neural network 
  - Test with some predictions
  - Save the model in a binary file
"""

import sys, os , requests, random
import numpy as np
from helpers.utils import *
from halo import Halo


# Adding the module path to the sys path 
so_dir = add_module_path_to_sys_path(__file__)

import NeuralNetPy as NNP

network = NNP.models.Network()

network.addLayer(NNP.layers.Dense(15, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.GLOROT))
network.addLayer(NNP.layers.Dense(20, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.GLOROT))
network.addLayer(NNP.layers.Dense(10, NNP.ACTIVATION.SOFTMAX, NNP.WEIGHT_INIT.LECUN))

# Setting up the networks parameters
network.setup(optimizer=NNP.optimizers.SGD(1), epochs=1, loss=NNP.LOSS.MCE)

inputs = list()

labels = [random.randint(0, 9) for i in range(10)]

for label in labels : 
  print(label)

for i in range(10):
  inputs.append([random.uniform(0, 1) for z in range(15)])

training_data = NNP.TrainingData2dI(inputs, labels)

training_data.batch(5)

network.train(training_data)