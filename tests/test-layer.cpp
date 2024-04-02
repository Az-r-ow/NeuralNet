#include <Eigen/Dense>
#include <Network.hpp>
#include <catch2/catch_test_macros.hpp>
#include <layers/Dropout.hpp>
#include <utils/Functions.hpp>
#include <vector>

using namespace NeuralNet;

TEST_CASE("Flatten's flatten() works as expected", "[layer]") {
  Flatten flattenLayer({2, 3});

  std::vector<std::vector<std::vector<double>>> inputs = {
      {{1, 2, 3}, {4, 5, 6}}, {{6, 5, 4}, {3, 2, 1}}};

  Eigen::MatrixXd expected(2, 6);

  expected << 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1;

  Eigen::MatrixXd outputs = flattenLayer.flatten(inputs);

  REQUIRE(outputs == expected);
};

TEST_CASE("Dropout layer", "[layer]") {
  double rate = 0.5;
  Dropout dropoutLayer = Dropout(rate);

  WHEN("Small inputs matrix") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Constant(4, 4, 1);

    Eigen::MatrixXd outputs = dropoutLayer.feedInputs(inputs);

    int count = 0;
    for (int i = 0; i < outputs.rows(); i++) {
      for (int j = 0; j < outputs.cols(); j++) {
        if (outputs(i, j) == 0) {
          count++;
        }
      }
    }

    REQUIRE(count ==
            static_cast<int>((1 - rate) * inputs.rows() * inputs.cols()));

    // Test scale factor
    CHECK(inputs.sum() == outputs.sum());
  }

  WHEN("Large inputs matrix") {
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Constant(30, 30, 1);

    Eigen::MatrixXd outputs = dropoutLayer.feedInputs(inputs);

    int count = 0;
    for (int i = 0; i < outputs.rows(); i++) {
      for (int j = 0; j < outputs.cols(); j++) {
        if (outputs(i, j) == 0) {
          count++;
        }
      }
    }

    REQUIRE(count == static_cast<int>(rate * inputs.rows() * inputs.cols()));

    // Test scale factor
    CHECK(inputs.sum() == outputs.sum());
  }
}