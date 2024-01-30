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
    CHECK(batches[0] == std::vector<int>{1, 2});
    CHECK(batches[1] == std::vector<int>{3, 4});
  };

  GIVEN("A vector of vectors of ints")
  {
    std::vector<std::vector<int>> data = {
        {1, 2, 3},
        {1, 3, 4},
        {1, 3, 5},
        {1, 3, 4}};

    NeuralNet::Tensor t(data);

    // Creating batches of 3
    t.batch(2);

    auto batches = t.getBatchedData();

    CHECK(batches.size() == 2);
    CHECK(batches[0] == std::vector<std::vector<int>>{{1, 2, 3}, {1, 3, 4}});
    CHECK(batches[1] == std::vector<std::vector<int>>{{1, 3, 5}, {1, 3, 4}});
  }
}