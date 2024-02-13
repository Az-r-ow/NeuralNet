#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <Eigen/Dense>
#include <Network.hpp>
#include <utils/Functions.hpp>

using namespace NeuralNet;

SCENARIO("Flatten's flatten() works as expected")
{
  Flatten flattenLayer({2, 3});

  std::vector<std::vector<std::vector<double>>> inputs = {
      {{1, 2, 3}, {4, 5, 6}},
      {{6, 5, 4}, {3, 2, 1}}};

  Eigen::MatrixXd expected(2, 6);

  expected << 1, 2, 3, 4, 5, 6,
      6, 5, 4, 3, 2, 1;

  Eigen::MatrixXd outputs = flattenLayer.flatten(inputs);

  REQUIRE(outputs == expected);
};