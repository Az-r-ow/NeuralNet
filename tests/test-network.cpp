#include <catch2/catch_test_macros.hpp>
#include <src/NeuralNet/Network.hpp>

using namespace NeuralNet;

SCENARIO("Layers are initialized correctly in the network")
{
  GIVEN("An empty network")
  {
    Network network;

    THEN("Number layer == 0")
    {
      REQUIRE(network.getNumLayers() == 0);
    }

    WHEN("3 layers are added")
    {
      Layer layer1 = Layer(2, RELU, GLOROT);
      Layer layer2 = Layer(3, RELU, GLOROT);
      Layer layer3 = Layer(1, RELU, GLOROT);

      network.addLayer(layer1);
      network.addLayer(layer2);
      network.addLayer(layer3);

      THEN("Number layer == 3")
      {
        REQUIRE(network.getNumLayers() == 3);
      }

      THEN("Right number neurons in layers")
      {
        CHECK(layer1.getNumNeurons() == 2);
        CHECK(layer2.getNumNeurons() == 3);
      }

      THEN("Weights matrices have correct sizes")
      {
        MatrixXd weightsL2 = layer2.getWeights();
        MatrixXd weightsL3 = layer3.getWeights();

        /**
         * The number of rows should be equal to
         * the number of neurons in the previous layer.
         *
         * The number of columns should be equal to
         * the number of neurons in the current layer.
         */
        REQUIRE(weightsL2.cols() == layer2.getNumNeurons());
        REQUIRE(weightsL3.rows() == layer2.getNumNeurons());
        REQUIRE(weightsL3.cols() == layer3.getNumNeurons());
        REQUIRE(weightsL2.rows() == layer1.getNumNeurons());
      }

      // todo: Check that weight values are initialized correctly
      // THEN("Weights initialized correctly")
      // {
      // }
    }
  }
}