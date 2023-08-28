import sys, os , requests
import numpy as np
from utils import *
from halo import Halo

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

network = NNP.Network()

network.addLayer(NNP.Layer(3))
network.addLayer(NNP.Layer(4))
network.addLayer(NNP.Layer(5))


print(network.getNumLayers())

layer1 = network.getLayer(0)

print(layer1.getNumNeurons())

# Remove sys.path modification
sys.path.remove(so_dir)