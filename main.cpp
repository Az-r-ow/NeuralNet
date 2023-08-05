#include "main.hpp"

int main(int argc, char *argv[])
{
   Network network;
   Layer layer1 = Layer(3, RELU, GLOROT);
   Layer layer2 = Layer(2, RELU, GLOROT);
   Layer layerOuput = Layer(2, RELU, GLOROT);

   network.addLayer(layer1);
   network.addLayer(layer2);
   network.addLayer(layerOuput);

   // training the network
   vector<vector<double>> inputs = {{0, 0, 0}};
   vector<double> labels = {1};
   network.train(inputs, labels);

   Layer input = network.getLayer(0);
   Layer test = network.getLayer(1);
   Layer test2 = network.getLayer(2);

   std::cout << "Input Layer : " << std::endl;
   input.printWeights();
   input.printOutputs();

   std::cout << "Layer 2 : "
             << std::endl;
   test.printWeights();
   test.printOutputs();

   std::cout << "Output Layer : "
             << std::endl;
   test2.printWeights();
   test2.printOutputs();

   return 0;
}