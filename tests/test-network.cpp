#include <Eigen/Dense>
#include <Network.hpp>
#include <catch2/catch_test_macros.hpp>
#include <utils/Functions.hpp>

#include "test-macros.hpp"

using namespace NeuralNet;

SCENARIO("Basic small network functions") {
  GIVEN("A small neural network") {
    Network sn;  // sn - small network
    std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);

    sn.setup(optimizer, LOSS::QUADRATIC);

    std::shared_ptr<Layer> layer1 =
        std::make_shared<Dense>(2, ACTIVATION::RELU, WEIGHT_INIT::HE);
    std::shared_ptr<Layer> layer2 =
        std::make_shared<Dense>(2, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);

    sn.addLayer(layer1);
    sn.addLayer(layer2);

    WHEN("Network trained") {
      std::vector<std::vector<double>> trainingInputs = {{1.0, 1.0}};
      std::vector<double> label = {1};

      Eigen::MatrixXd preTrainWeights = sn.getLayer(1)->getWeights();

      sn.train(trainingInputs, label);

      CHECK(sn.getLayer(1)->getWeights() != preTrainWeights);
    }
  }
}

SCENARIO("Layers are initialized correctly in the network") {
  GIVEN("An empty network") {
    Network network;
    std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);
    // Setting up the parameters
    network.setup(optimizer, LOSS::QUADRATIC);

    THEN("Number layers == 0") { REQUIRE(network.getNumLayers() == 0); }

    WHEN("3 layers are added") {
      std::shared_ptr<Layer> layer1 =
          std::make_shared<Layer>(2, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
      std::shared_ptr<Layer> layer2 =
          std::make_shared<Layer>(3, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
      std::shared_ptr<Layer> layer3 =
          std::make_shared<Layer>(1, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);

      network.addLayer(layer1);
      network.addLayer(layer2);
      network.addLayer(layer3);

      THEN("Number layer == 3") { REQUIRE(network.getNumLayers() == 3); }

      THEN("Right number neurons in layers") {
        CHECK(layer1->getNumNeurons() == 2);
        CHECK(layer2->getNumNeurons() == 3);
      }

      THEN("Weights matrices have correct sizes") {
        Eigen::MatrixXd weightsL2 = layer2->getWeights();
        Eigen::MatrixXd weightsL3 = layer3->getWeights();

        /**
         * The number of rows should be equal to
         * the number of neurons in the previous layer
         *
         * The number of columns should be equal to
         * the number of neurons in the current layer.
         */
        REQUIRE(weightsL2.cols() == layer2->getNumNeurons());
        REQUIRE(weightsL3.rows() == layer2->getNumNeurons());
        REQUIRE(weightsL3.cols() == layer3->getNumNeurons());
        REQUIRE(weightsL2.rows() == layer1->getNumNeurons());
      }

      THEN("Weights are not initialized to 0") {
        Eigen::MatrixXd weightsL2 = layer2->getWeights();
        Eigen::MatrixXd weightsL3 = layer3->getWeights();

        REQUIRE(weightsL2 !=
                Eigen::MatrixXd::Zero(weightsL2.rows(), weightsL2.cols()));
        REQUIRE(weightsL3 !=
                Eigen::MatrixXd::Zero(weightsL3.rows(), weightsL3.cols()));
      }
    }
  }
}

SCENARIO("The network remains the same when trained with null inputs") {
  // Limiting the network with 1 epoch
  Network network;
  std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);
  // Setting up the parameters
  network.setup(optimizer, LOSS::QUADRATIC);

  std::shared_ptr<Layer> inputLayer =
      std::make_shared<Layer>(2, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
  std::shared_ptr<Layer> hiddenLayer =
      std::make_shared<Layer>(3, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);
  std::shared_ptr<Layer> outputLayer =
      std::make_shared<Layer>(1, ACTIVATION::RELU, WEIGHT_INIT::GLOROT);

  network.addLayer(inputLayer);
  network.addLayer(hiddenLayer);
  network.addLayer(outputLayer);

  GIVEN("A network with 3 layers") {
    THEN("Number of layers = 3") { REQUIRE(network.getNumLayers() == 3); }

    WHEN("Null inputs are passed") {
      std::vector<std::vector<double>> nullInputs = {{0, 0}};
      std::vector<double> labels = {0};

      // caching the weights before training for later comparison
      Eigen::MatrixXd preTrainW1 = network.getLayer(1)->getWeights();
      Eigen::MatrixXd preTrainW2 = network.getLayer(2)->getWeights();

      // Training with null weights and inputs
      network.train(nullInputs, labels, 2);

      THEN("Outputs are 0") {
        Eigen::MatrixXd outputs = network.getOutputLayer()->getOutputs();

        REQUIRE(outputs ==
                Eigen::MatrixXd::Zero(outputs.rows(), outputs.cols()));
      }

      AND_THEN("The weights remain the same") {
        CHECK(network.getLayer(1)->getWeights() == preTrainW1);
        CHECK(network.getLayer(2)->getWeights() == preTrainW2);
      }
    }
  }
}

SCENARIO("The network updates the weights and biases as pre-calculated") {
  Network network;
  std::shared_ptr<Optimizer> sgdOptimizer = std::make_shared<SGD>(1.5);

  network.setup(sgdOptimizer, LOSS::QUADRATIC);

  std::shared_ptr<Layer> inputLayer =
      std::make_shared<Layer>(3, ACTIVATION::RELU);
  std::shared_ptr<Layer> hiddenLayer =
      std::make_shared<Layer>(3, ACTIVATION::RELU, WEIGHT_INIT::CONSTANT);
  std::shared_ptr<Layer> outputLayer =
      std::make_shared<Layer>(2, ACTIVATION::SIGMOID, WEIGHT_INIT::CONSTANT);

  network.addLayer(inputLayer);
  network.addLayer(hiddenLayer);
  network.addLayer(outputLayer);

  std::vector<std::vector<double>> inputs = {
      {0.7, 0.3, 0.1}, {0.5, 0.3, 0.1}, {1.0, 0.2, 0.4}, {-0.5, 0.3, -1}};

  std::vector<double> labels = {1, 1, 0, 1};

  WHEN("Predicting without training") {
    Eigen::MatrixXd predictions = network.predict(inputs);
    Eigen::MatrixXd expectedPredictions(4, 2);

    expectedPredictions << 0.96442881, 0.96442881, 0.93702664, 0.93702664,
        0.99183743, 0.99183743, 0.5, 0.5;

    CHECK_MATRIX_APPROX(predictions, expectedPredictions, EPSILON);
  }

  TrainingData trainData(inputs, labels);

  trainData.batch(2);

  network.train(trainData);

  std::shared_ptr<Layer> oLayer = network.getLayer(2);
  std::shared_ptr<Layer> hLayer = network.getLayer(1);

  // Expected Weights Hidden Layer
  Eigen::MatrixXd EWHL(3, 3);

  EWHL << 0.90743867, 0.90743867, 0.90743867, 0.9583673, 0.9583673, 0.9583673,
      0.97931547, 0.97931547, 0.97931547;

  // Expected Biases Hidden Layer
  Eigen::MatrixXd EBHL(1, 3);

  EBHL << -0.14558262, -0.14558262, -0.14558262;

  // Expected Weights Output Layer
  Eigen::MatrixXd EWOL(3, 2);

  EWOL << 0.87250736, 0.97733271, 0.87250736, 0.97733271, 0.87250736,
      0.97733271;

  // Expected Biases Output Layer
  Eigen::MatrixXd EBOL(1, 2);

  EBOL << -0.30563573, 0.17284546;

  CHECK_MATRIX_APPROX(hLayer->getWeights(), EWHL, EPSILON);
  CHECK_MATRIX_APPROX(hLayer->getBiases(), EBHL, EPSILON);
  CHECK_MATRIX_APPROX(oLayer->getWeights(), EWOL, EPSILON);
  CHECK_MATRIX_APPROX(oLayer->getBiases(), EBOL, EPSILON);

  WHEN("Predicting after training") {
    Eigen::MatrixXd predictions = network.predict(inputs);
    Eigen::MatrixXd expectedPredictions(4, 2);

    expectedPredictions << 0.87919928, 0.93926274, 0.8190347, 0.90082421,
        0.96141716, 0.98396999, 0.42418036, 0.54310411;

    CHECK_MATRIX_APPROX(predictions, expectedPredictions, EPSILON);
  }
}
