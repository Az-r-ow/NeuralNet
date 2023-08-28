import sys, os 

script_dir = os.path.dirname(os.path.abspath(__file__))

so_dir = os.path.join(script_dir, "..", "..", "build")

# Adding the path to the build dir to the sys.path
sys.path.append(so_dir)

import NeuralNetPy as NNP

network = NNP.Network()

network.addLayer(NNP.Layer(3))
network.addLayer(NNP.Layer(4))
network.addLayer(NNP.Layer(5))


print(network.getNumLayers())

layer1 = network.getLayer(0)

print(layer1.getNumNeurons())

# Remove sys.path modification
sys.path.remove(so_dir)