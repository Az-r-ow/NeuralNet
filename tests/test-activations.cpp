#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <activations/Relu.hpp>
#include <activations/Sigmoid.hpp>
#include <activations/Softmax.hpp>
#include "helper-functions.hpp"

using Eigen::MatrixXd;

const double EPSILON = 0.001;

TEST_CASE("Relu activates correctly", "[function]")
{
  MatrixXd inputs(4, 1);

  MatrixXd expectedOutputs(4, 1);

  inputs << -1,
      0,
      4,
      10;

  expectedOutputs << 0,
      0,
      4,
      10;

  CHECK(NeuralNet::Relu::activate(inputs) == expectedOutputs);
}

TEST_CASE("Relu differentiate correctly", "[function]")
{
  const int num_cols = 2;
  const int num_rows = 2;

  MatrixXd test_m1 = MatrixXd::Random(num_rows, num_cols);
  MatrixXd expected_m1(num_rows, num_cols);

  for (int r = 0; r < num_rows; r++)
  {
    for (int c = 0; c < num_cols; c++)
    {
      expected_m1(r, c) = test_m1(r, c) > 0 ? 1 : 0;
    }
  }

  CHECK(NeuralNet::Relu::diff(test_m1) == expected_m1);
}

TEST_CASE("Sigmoid Activates correctly", "[function]")
{
  MatrixXd inputs(4, 1);

  MatrixXd expectedOutputs(4, 1);

  inputs << -7.66,
      -1,
      0,
      10;

  // Pre-calculated
  expectedOutputs << 0,
      0.268,
      0.5,
      1;

  CHECK_MATRIX_APPROX(NeuralNet::Sigmoid::activate(inputs), expectedOutputs, EPSILON);
}

TEST_CASE("Sigmoid differentiates correctly", "[function]")
{
  const int num_rows = 2;
  const int num_cols = 2;

  MatrixXd test_m1 = MatrixXd::Random(num_rows, num_cols);
  MatrixXd expected_m1(num_rows, num_cols);

  for (int r = 0; r < num_rows; r++)
  {
    for (int c = 0; c < num_cols; c++)
    {
      expected_m1(r, c) = test_m1(r, c) * (1 - test_m1(r, c));
    }
  }

  CHECK(NeuralNet::Sigmoid::diff(test_m1) == expected_m1);
}

TEST_CASE("Softmax activates correctly", "[function]")
{
  /**
   * Expected outputs should be updated because Softmax now scales the inputs
   */
  MatrixXd inputs(4, 1);

  MatrixXd expectedOutputs(4, 1);

  inputs << -2, 0, 1, 2;

  expectedOutputs << 0.012, 0.0889, 0.241, 0.657;

  CHECK_MATRIX_APPROX(NeuralNet::Softmax::activate(inputs), expectedOutputs, EPSILON);
}

TEST_CASE("Softmax differentiates correctly", "[function]")
{
  MatrixXd inputs(4, 1);

  MatrixXd expectedOutputs(4, 1);

  inputs << 1, 2, 3, 4;

  expectedOutputs << 0.031, 0.0795, 0.18, 0.2292;

  MatrixXd activatedInputs = NeuralNet::Softmax::activate(inputs);

  CHECK_MATRIX_APPROX(NeuralNet::Softmax::diff(activatedInputs), expectedOutputs, EPSILON);
}