#include "main.hpp"

int main(int argc, char *argv[])
{
   Network network;
   Layer layer(3, RELU, GLOROT);
   Layer layer2(4, RELU, GLOROT);

   network.addLayer(layer);
   network.addLayer(layer2);
   network.addLayer(Layer(5, RELU, GLOROT));

   Layer test = network.getLayer(1);
   Layer test2 = network.getLayer(2);

   vector<vector<double>> inputs = {{2, 3, 4}};
   vector<double> labels = {2, 2, 3};

   network.train(inputs, labels);

   std::cout << "Layer 1 : \n"
             << std::endl;
   test.printOutputs();

   std::cout << test.getOutputs() << std::endl;

   std::cout << "Layer 2 : \n"
             << std::endl;
   test2.printOutputs();

   return 0;
}