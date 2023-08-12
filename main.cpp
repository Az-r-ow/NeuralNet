#include "main.hpp"

using namespace NeuralNet;

int main(int argc, char *argv[])
{
   Network network;
   Layer layer1 = Layer(3, RELU, GLOROT);
   Layer layer2 = Layer(2, RELU, GLOROT);
   Layer layerOuput = Layer(2, RELU, GLOROT);

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
   vector<vector<double>> inputs = {{1, 1, 1}};
   vector<double> labels = {1};
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