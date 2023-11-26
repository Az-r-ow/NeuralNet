"""
  In this file, we :
  - Download the mnist database if not downloaded already
  - Train the neural network 
  - Test with some predictions
  - Save the model in a binary file
"""

import sys, os , requests, random
import numpy as np
from utils import *
from halo import Halo

NUM_TRAININGS = 10
NUM_PREDICTIONS = 1000
MNIST_DATASET_FILE = "./dataset/mnist.npz"

# Adding the module path to the sys path 
so_dir = add_module_path_to_sys_path(__file__)

import NeuralNetPy as NNP

# If file doesn't exists create and download the data
if not file_exists(MNIST_DATASET_FILE):
  print("Mnist dataset not found")
  get_MNIST_dataset(MNIST_DATASET_FILE)

# Otherwise load data from file
(x_train, y_train), (x_test, y_test) = load_data(MNIST_DATASET_FILE)

network = NNP.Network()

network.addLayer(NNP.Flatten((28, 28)))
network.addLayer(NNP.Dense(128, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
network.addLayer(NNP.Dense(10, NNP.ACTIVATION.SOFTMAX, NNP.WEIGHT_INIT.LECUN))

# Setting up the networks parameters
network.setup(optimizer=NNP.SGD(0.01), epochs=1, loss=NNP.LOSS.MCE)

# combining the data with the labels for later shuffling 
combined = list(zip(x_train, y_train))

# shuffling the combined list 
random.shuffle(combined)

# separating them 
x_train, y_train = zip(*combined)

# preparing the training data
f_x_train = [normalize_img(x) for x in x_train]

trainingData = NNP.TrainingData3dI(f_x_train[:NUM_TRAININGS], y_train[:NUM_TRAININGS])

trainingData.batch(5)

network.train(trainingData)

f_x_test = [normalize_img(x) for x in x_test]

# preparing the testing data
predictions = network.predict(f_x_test[:NUM_PREDICTIONS])

predicted_numbers = find_highest_indexes_in_matrix(predictions)

predicted_numbers = [i + 1 for i in predicted_numbers]

(accuracy, n, correct) = get_accuracy(predicted_numbers, y_test)

# Getting the prediction's accuracy 
print(f"Num correct predictions : {correct}/{n} - accuracy {accuracy}")

# save trained model to file 
NNP.Model.save_to_file("model.bin",  network)

# Remove sys.path modification
sys.path.remove(so_dir)