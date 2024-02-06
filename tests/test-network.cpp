#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <Network.hpp>
#include <utils/Functions.hpp>
#include "test-macros.hpp"

using namespace NeuralNet;

SCENARIO("Basic small network functions")
{
  GIVEN("A small neural network")
  {
    Network sn; // sn - small network
    std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);

    sn.setup(optimizer, 1, LOSS::QUADRATIC);

    std::shared_ptr<Layer> layer1 = std::make_shared<Dense>(2, ACTIVATION::RELU, WEIGHT_INIT::HE);
    std::shared_ptr<Layer> layer2 = std::make_shared<Dense>(2, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);

    sn.addLayer(layer1);
    sn.addLayer(layer2);

    WHEN("Network trained")
    {
      std::vector<std::vector<double>> trainingInputs = {{1.0, 1.0}};
      std::vector<double> label = {1};

      Eigen::MatrixXd preTrainWeights = sn.getLayer(1)->getWeights();

      sn.train(trainingInputs, label);

      CHECK(sn.getLayer(1)->getWeights() != preTrainWeights);
    }
  }
}

SCENARIO("Layers are initialized correctly in the network")
{
  GIVEN("An empty network")
  {
    Network network;
    std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);
    // Setting up the parameters
    network.setup(optimizer, 1, LOSS::QUADRATIC);

    THEN("Number layers == 0")
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

SCENARIO("The network updates the weights and biases as pre-calculated")
{
  Network network;
  std::shared_ptr<Optimizer> sgdOptimizer = std::make_shared<SGD>(1.5);

  network.setup(sgdOptimizer, 1, LOSS::QUADRATIC);

  std::shared_ptr<Layer> inputLayer = std::make_shared<Layer>(3, ACTIVATION::RELU);
  std::shared_ptr<Layer> outputLayer = std::make_shared<Layer>(2, ACTIVATION::SIGMOID, WEIGHT_INIT::CONSTANT);

  network.addLayer(inputLayer);
  network.addLayer(outputLayer);

  std::vector<std::vector<double>> inputs = {
      {0.7, 0.3, 0.1},
      {0.5, 0.3, 0.1},
      {1.0, 0.2, 0.4},
      {-0.5, 0.3, -1}};

  std::vector<double> labels = {1, 1, 0, 1};

  WHEN("Predicting without training")
  {
    Eigen::MatrixXd predictions = network.predict(inputs);
    Eigen::MatrixXd expectedPredictions(4, 2);

    expectedPredictions << 0.75026, 0.75026,
        0.71095, 0.71095,
        0.832018, 0.832018,
        0.231475, 0.231475;

    CHECK_MATRIX_APPROX(predictions, expectedPredictions, EPSILON);
  }

  TrainingData trainData(inputs, labels);

  trainData.batch(2);

  network.train(trainData);

  std::shared_ptr<Layer> oLayer = network.getLayer(1);

  Eigen::MatrixXd expectedWeights(3, 2);

  expectedWeights << 0.855486, 0.837484,
      0.877136, 1.08112,
      1.03485, 0.744304;

  Eigen::MatrixXd expectedBiases(1, 2);

  expectedBiases << -0.378823, 0.220239;

  CHECK_MATRIX_APPROX(oLayer->getWeights(), expectedWeights, EPSILON);
  CHECK_MATRIX_APPROX(oLayer->getBiases(), expectedBiases, EPSILON);
}
