#include "main.hpp"

int main(int argc, char *argv[])
{
   Network network;
   Layer layer(3, RELU, GLOROT);
   Layer layer2(4, RELU, GLOROT);
   Layer layer3(5, RELU, GLOROT);

   network.addLayer(layer);
   network.addLayer(layer2);
   network.addLayer(layer3);

   Layer input = network.getLayer(0);
   Layer test = network.getLayer(1);
   Layer test2 = network.getLayer(2);

   std::cout << "Input Layer : " << std::endl;
   input.printWeights();
   input.printOutputs();

   std::cout << "Layer 1 : "
             << std::endl;
   test.printWeights();
   test.printOutputs();

   std::cout << "Layer 2 : "
             << std::endl;
   test2.printWeights();
   test2.printOutputs();

   return 0;
}