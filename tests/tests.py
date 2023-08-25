import sys 
import os 

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the directory containing the .so file
so_dir = os.path.join(script_dir, "..", "build")

# Add the .so directory to sys.path temporarily
sys.path.append(so_dir)

import NeuralNetPy as NNP

network = NNP.Network()

network.addLayer(NNP.Layer(3))
network.addLayer(NNP.Layer(4))
network.addLayer(NNP.Layer(5))


print(network.getNumLayers())

layer1 = network.getLayer(0)

print(layer1.getNumNeurons())