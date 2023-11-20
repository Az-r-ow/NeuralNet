#include <catch2/catch_test_macros.hpp>
#include <data/Tensor.hpp>

using namespace NeuralNet;

SCENARIO("Tensor batches data correctly")
{
  GIVEN("A vector of ints")
  {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7};

    Tensor t(data);

    // Creating batches of 2 elements
    t.batch(2);

    auto batches = t.getBatchedData();

    CHECK(batches.size() == 4);
  }
}