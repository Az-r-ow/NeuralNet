#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <utils/Functions.hpp>
#include "test-macros.hpp"

using namespace Catch::Matchers;
using namespace NeuralNet;

const double ERR_MARGIN = 0.001;

TEST_CASE("Sqr function returns the right square", "[helper_function]")
{
  CHECK(sqr(0) == 0);
  CHECK(sqr(2) == 4);
  CHECK(sqr(10) == 100);
  REQUIRE_THAT(sqr(7.8877), WithinAbs(62.215, ERR_MARGIN));
  CHECK(sqr(657666) == 432524567556);
}

TEST_CASE("vectorToMatrixXd outputs the correct format", "[helper_function]")
{
  std::vector<std::vector<double>> v = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9}};

  Eigen::MatrixXd expected(3, 3);

  expected << 1, 2, 3,
      4, 5, 6,
      7, 8, 9;

  CHECK(vectorToMatrixXd(v) == expected);
}

TEST_CASE("randomWeightInit initializes value properly", "[weight_initialization]")
{
  Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(5, 5);
  double min = -2.0;
  double max = 2.0;

  randomWeightInit(&weights, min, max);
  CHECK_MATRIX_VALUES_IN_RANGE(weights, min, max);
}

TEST_CASE("randomDistMatrixInit initializes values properly", "[weight_initialization]")
{
  Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(5, 5);

  randomDistMatrixInit(&weights, -1.0, 3.2);

  // todo: test if random distribution init works
}