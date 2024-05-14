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