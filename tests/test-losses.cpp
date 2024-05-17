#include <activations/activations.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <losses/losses.hpp>

#include "test-macros.hpp"

using namespace Catch::Matchers;
using namespace NeuralNet;

TEST_CASE("Testing Binary Cross-Entropy loss with random values", "[losses]") {
  Eigen::MatrixXd o = Eigen::MatrixXd::Random(2, 2).array().abs();
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(2, 2);

  double loss = BCE::cmpLoss(o, y);

  REQUIRE(loss >= 0);
}

TEST_CASE("Testing Binary Cross-Entropy derivation with pre-calculated values",
          "[losses]") {
  Eigen::MatrixXd o(2, 2);
  Eigen::MatrixXd y(2, 2);

  o << 0.5, 0.5, 0.2, 0.8;
  y << 0, 1, 0, 1;

  Eigen::MatrixXd grad = BCE::cmpLossGrad(o, y);

  Eigen::MatrixXd exp(2, 2);

  exp << 2.0, -2.0, 1.25, -1.25;

  CHECK_MATRIX_APPROX(grad, exp, EPSILON);
}

TEST_CASE("Testing Binary Cross-Entropy with softmax activation", "[losses]") {
  Eigen::MatrixXd i = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(2, 2);

  y(0, 0) = 1;
  y(1, 1) = 1;

  Eigen::MatrixXd prob = Softmax::activate(i);

  double loss = BCE::cmpLoss(prob, y);

  CHECK(loss >= 0);

  Eigen::MatrixXd grad = BCE::cmpLossGrad(prob, y);

  bool hasNaN = grad.array().isNaN().any();

  CHECK_FALSE(hasNaN);
}