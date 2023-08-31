#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <Network.hpp>
#include <utils/Functions.hpp>

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
      Layer layer1 = Layer(2, ActivationName::RELU, WeightInit::GLOROT);
      Layer layer2 = Layer(3, ActivationName::RELU, WeightInit::GLOROT);
      Layer layer3 = Layer(1, ActivationName::RELU, WeightInit::GLOROT);

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

      THEN("Weights are not initialized to 0")
      {
        MatrixXd weightsL2 = layer2.getWeights();
        MatrixXd weightsL3 = layer3.getWeights();

        REQUIRE(weightsL2 != MatrixXd::Zero(weightsL2.rows(), weightsL2.cols()));
        REQUIRE(weightsL3 != MatrixXd::Zero(weightsL3.rows(), weightsL3.cols()));
      }
    }
  }
}

SCENARIO("The network back propagates")
{
  Network network;

  Layer inputLayer = Layer(3, ActivationName::SIGMOID);
  Layer hiddenLayer = Layer(2, ActivationName::SIGMOID);
  Layer outputLayer = Layer(1, ActivationName::SIGMOID);

  network.addLayer(inputLayer);
  network.addLayer(hiddenLayer);
  network.addLayer(outputLayer);

  GIVEN("A network with 3 layers")
  {
    THEN("Number of layers = 3")
    {
      REQUIRE(network.getNumLayers() == 3);
    }

    WHEN("Null inputs are passed")
    {
      std::vector<std::vector<double>> nullInputs = {{0, 0, 0}};
      std::vector<double> labels = {0};

      // caching the weights before training for later comparison
      MatrixXd preTrainW1 = network.getLayer(1).getWeights();
      MatrixXd preTrainW2 = network.getLayer(2).getWeights();

      // Training with null weights and inputs
      network.train(nullInputs, labels);

      THEN("Outputs are 0")
      {
        MatrixXd outputs = network.getOutputLayer().getOutputs();

        REQUIRE(outputs == MatrixXd::Zero(outputs.rows(), outputs.cols()));
      }

      AND_THEN("The weights remain the same")
      {
        CHECK(network.getLayer(1).getWeights() == preTrainW1);
        CHECK(network.getLayer(2).getWeights() == preTrainW2);
      }
    }

    WHEN("Inputs with random values are passed")
    {
      std::vector<std::vector<double>> randInputs;
      randInputs.push_back(randDVector(network.getLayer(0).getNumNeurons()));
      std::vector<double> labels = {0};

      // Caching weights before training for later comparison
      MatrixXd preTrainW1 = network.getLayer(1).getWeights();
      MatrixXd preTrainW2 = network.getLayer(2).getWeights();

      // Training with random values
      network.train(randInputs, labels);

      AND_THEN("The weights differ")
      {
        CHECK(network.getLayer(1).getWeights() != preTrainW1);
        CHECK(network.getLayer(2).getWeights() != preTrainW2);
      }
    }
  }
}
