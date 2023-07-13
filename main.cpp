#include "main.hpp"

int main(int argc, char *argv[])
{
   Network network;
   Layer layer(3);
   Layer layer2(4);

   network.addLayer(layer);
   network.addLayer(layer2);

   Layer test = network.getLayer(1);

   test.printWeights();

   return 0;
}