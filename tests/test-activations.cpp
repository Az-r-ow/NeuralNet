#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <activations/Relu.hpp>
#include <activations/Sigmoid.hpp>

using Eigen::MatrixXd;

const double ERR_MARGIN = 0.001;

const std::vector<double> testCases = {-200, -10, -7.66, -1, 0, 2, 10, 15.7689, 200};
// Test results for Relu activate function
const std::vector<double> testResultsReAct = {0, 0, 0, 0, 0, 2, 10, 15.768, 200};
// Test results for Sigmoid activate function
const std::vector<double> testResultsSigAct = {0, 0, 0, 0.268, 0.5, 0.88, 1, 1, 1};

TEST_CASE("Relu activates correctly")
{
  assert(testCases.size() == testResultsReAct.size());

  for (int i = 0; i < testCases.size(); i++)
  {
    CHECK_THAT(NeuralNet::Relu::activate(testCases[i]), Catch::Matchers::WithinAbs(testResultsReAct[i], ERR_MARGIN));
  }
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

TEST_CASE("Sigmoid Activates correctly")
{
  assert(testCases.size() == testResultsSigAct.size());

  for (int i = 0; i < testCases.size(); i++)
  {
    CHECK_THAT(NeuralNet::Sigmoid::activate(testCases[i]), Catch::Matchers::WithinAbs(testResultsSigAct[i], ERR_MARGIN));
  }
}