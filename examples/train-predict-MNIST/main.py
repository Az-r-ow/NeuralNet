import sys, os , requests, random
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from halo import Halo

NUM_TESTS = 5000
NUM_PREDICTIONS = 100

"""
  In the first few lines we add the build folder to the sys.paths 
  to be able to import the NeuralNet module from the generated .so file
"""

# Getting the path to the .so file 
script_dir = os.path.dirname(os.path.abspath(__file__))

so_dir = os.path.join(script_dir, "..", "..", "build")

# Adding the path to the build dir to the sys.path
sys.path.append(so_dir)

import NeuralNetPy as NNP

mnist_dataset_file = "./dataset/mnist.npz"
mnist_dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

spinner = Halo(text="Downloading the dataset", spinner="dots")

# If file doesn't exists create and download the data
if not os.path.exists(mnist_dataset_file):
  print(f"{mnist_dataset_file} not found")
  spinner.start()
  response = requests.get(mnist_dataset_url)
  spinner.stop()
  with open(mnist_dataset_file, 'wb') as f:
    f.write(response.content)

# Otherwise load data from file
(x_train, y_train), (x_test, y_test) = load_data(mnist_dataset_file)

network = NNP.Network(epochs=2, alpha=0.01, loss=NNP.LOSS.MCE)

network.setBatchSize(1)

network.addLayer(NNP.Layer(784))
network.addLayer(NNP.Layer(128, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
network.addLayer(NNP.Layer(10, NNP.ACTIVATION.SOFTMAX, NNP.WEIGHT_INIT.LECUN))

# combining the data with the labels for later shuffling 
combined = list(zip(x_train, y_train))

# shuffling the combined list 
random.shuffle(combined)

# separating them 
x_train, y_train = zip(*combined)

# preparing the training data
f_x_train = [normalize_img(x.flatten()) for x in x_train]

network.train(f_x_train[:NUM_TESTS], y_train[:NUM_TESTS])

f_x_test = [normalize_img(x.flatten()) for x in x_test]

# preparing the testing data
predictions = network.predict(f_x_test[:NUM_PREDICTIONS])

(accuracy, n, correct) = get_accuracy(predictions, y_test)

# Getting the prediction's accuracy 
print(f"Num correct predictions : {correct}/{n} - accuracy {accuracy}")

# Remove sys.path modification
sys.path.remove(so_dir)