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
    std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);
    // Setting up the parameters
    network.setup(optimizer, 1, LOSS::QUADRATIC);

    THEN("Number layer == 0")
    {
      REQUIRE(network.getNumLayers() == 0);
    }

    WHEN("3 layers are added")
    {
      std::shared_ptr<Layer> layer1 = std::make_shared<Layer>(2, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
      std::shared_ptr<Layer> layer2 = std::make_shared<Layer>(3, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
      std::shared_ptr<Layer> layer3 = std::make_shared<Layer>(1, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);

      network.addLayer(layer1);
      network.addLayer(layer2);
      network.addLayer(layer3);

      THEN("Number layer == 3")
      {
        REQUIRE(network.getNumLayers() == 3);
      }

      THEN("Right number neurons in layers")
      {
        CHECK(layer1->getNumNeurons() == 2);
        CHECK(layer2->getNumNeurons() == 3);
      }

      THEN("Weights matrices have correct sizes")
      {
        Eigen::MatrixXd weightsL2 = layer2->getWeights();
        Eigen::MatrixXd weightsL3 = layer3->getWeights();

        /**
         * The number of rows should be equal to
         * the number of neurons in the previous layer->
         *
         * The number of columns should be equal to
         * the number of neurons in the current layer.
         */
        REQUIRE(weightsL2.cols() == layer2->getNumNeurons());
        REQUIRE(weightsL3.rows() == layer2->getNumNeurons());
        REQUIRE(weightsL3.cols() == layer3->getNumNeurons());
        REQUIRE(weightsL2.rows() == layer1->getNumNeurons());
      }

      THEN("Weights are not initialized to 0")
      {
        Eigen::MatrixXd weightsL2 = layer2->getWeights();
        Eigen::MatrixXd weightsL3 = layer3->getWeights();

        REQUIRE(weightsL2 != Eigen::MatrixXd::Zero(weightsL2.rows(), weightsL2.cols()));
        REQUIRE(weightsL3 != Eigen::MatrixXd::Zero(weightsL3.rows(), weightsL3.cols()));
      }
    }
  }
}

SCENARIO("The network remains the same when trained with null inputs")
{
  // Limiting the network with 1 epoch
  Network network;
  std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);
  // Setting up the parameters
  network.setup(optimizer, 1, LOSS::QUADRATIC);
  /**
   * Setting the batch size to 1 so that the network backpropagates on the first training sample
   */
  network.setBatchSize(1);

  std::shared_ptr<Layer> inputLayer = std::make_shared<Layer>(2, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
  std::shared_ptr<Layer> hiddenLayer = std::make_shared<Layer>(3, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
  std::shared_ptr<Layer> outputLayer = std::make_shared<Layer>(1, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);

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
      std::vector<std::vector<double>> nullInputs = {{0, 0}};
      std::vector<double> labels = {0};

      // caching the weights before training for later comparison
      Eigen::MatrixXd preTrainW1 = network.getLayer(1)->getWeights();
      Eigen::MatrixXd preTrainW2 = network.getLayer(2)->getWeights();

      // Training with null weights and inputs
      network.train(nullInputs, labels);

      THEN("Outputs are 0")
      {
        Eigen::MatrixXd outputs = network.getOutputLayer()->getOutputs();

        REQUIRE(outputs == Eigen::MatrixXd::Zero(outputs.rows(), outputs.cols()));
      }

      AND_THEN("The weights remain the same")
      {
        CHECK(network.getLayer(1)->getWeights() == preTrainW1);
        CHECK(network.getLayer(2)->getWeights() == preTrainW2);
      }
    }
  }
}

SCENARIO("The network back propagates")
{
  Network network;
  std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);
  // Setting up the parameters
  network.setup(optimizer, 1, LOSS::QUADRATIC);

  /**
   * Setting the batch size to 1 so that the network backpropagates on the first training sample
   */
  network.setBatchSize(1);

  std::shared_ptr<Layer> inputLayer = std::make_shared<Layer>(2, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
  std::shared_ptr<Layer> hiddenLayer = std::make_shared<Layer>(3, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
  std::shared_ptr<Layer> outputLayer = std::make_shared<Layer>(2, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);

  network.addLayer(inputLayer);
  network.addLayer(hiddenLayer);
  network.addLayer(outputLayer);

  GIVEN("Random inputs and feedback")
  {
    std::vector<std::vector<double>> randInputs;
    randInputs.push_back(randDVector(network.getLayer(0)->getNumNeurons()));

    std::vector<double> labels = {1};

    // Caching weights before training for later comparison
    Eigen::MatrixXd preTrainW1 = network.getLayer(1)->getWeights();
    Eigen::MatrixXd preTrainW2 = network.getLayer(2)->getWeights();

    // Training with random values
    network.train(randInputs, labels);

    THEN("The weights differ")
    {
      CHECK(network.getLayer(1)->getWeights() != preTrainW1);
      CHECK(network.getLayer(2)->getWeights() != preTrainW2);
    }
  }
}
