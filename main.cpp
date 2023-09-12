#include "main.hpp"

using namespace NeuralNet;

int main(int argc, char *argv[])
{
   Network network;
   Layer layer1 = Layer(3, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);
   Layer layer2 = Layer(2, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);
   Layer layerOuput = Layer(2, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);

   network.addLayer(layer1);
   network.addLayer(layer2);
   network.addLayer(layerOuput);

   std::cout << "Input Layer before training : "
             << "\n";
   layer1.printWeights();
   layer1.printOutputs();

   std::cout << "Layer 2 before training : "
             << "\n";
   layer2.printWeights();
   layer2.printOutputs();

   std::cout << "Output Layer before training : "
             << "\n";
   layerOuput.printWeights();
   layerOuput.printOutputs();

   // training the network
   std::vector<std::vector<double>> inputs;
   inputs.push_back(randDVector(layer1.getNumNeurons(), -1, 1));
   std::vector<double> labels = {1};
   network.train(inputs, labels);

   Layer input = network.getLayer(0);
   Layer test = network.getLayer(1);
   Layer test2 = network.getLayer(2);

   std::cout << "Input Layer after training : "
             << "\n";
   input.printWeights();
   input.printOutputs();

   std::cout << "Layer 2 after training : "
             << "\n";
   test.printWeights();
   test.printOutputs();

   std::cout << "Output Layer after training : "
             << "\n";
   test2.printWeights();
   test2.printOutputs();

   return 0;
}