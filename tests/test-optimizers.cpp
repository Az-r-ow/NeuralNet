#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <optimizers/optimizers.hpp>

using namespace NeuralNet;

SCENARIO("Testing SGD Optimizer") {
  SGD optimizer(1);

  GIVEN("Weights that initialized to 1") {
    Eigen::MatrixXd weights = Eigen::MatrixXd::Constant(2, 2, 1);

    GIVEN("Gradients equal to the weights") {
      Eigen::MatrixXd weightsGrad = Eigen::MatrixXd::Constant(2, 2, 1);

      optimizer.updateWeights(weights, weightsGrad);

      REQUIRE(weights == Eigen::MatrixXd::Zero(2, 2));
    };
  }
}