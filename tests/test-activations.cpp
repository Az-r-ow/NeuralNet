#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <activations/Relu.hpp>
#include <activations/Sigmoid.hpp>

using Eigen::MatrixXd;

TEST_CASE("Relu activates correctly")
{
  CHECK(NeuralNet::Relu::activate(-10) == 0);
  CHECK(NeuralNet::Relu::activate(0) == 0);
  CHECK(NeuralNet::Relu::activate(3) == 3);
  CHECK(NeuralNet::Relu::activate(200) == 200);
}

TEST_CASE("Relu differentiate correctly")
{
  int num_cols = 2;
  int num_rows = 2;

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