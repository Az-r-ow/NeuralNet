#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <Functions.hpp>

using namespace NeuralNet;

TEST_CASE("Relu of negative numbers always returns 0", "[function]")
{
  REQUIRE(relu(-1) == 0);
  REQUIRE(relu(-898.99) == 0);
  REQUIRE(relu(-0) == 0);
}

TEST_CASE("Relu of positive numbers always returns the number itself", "[function]")
{
  REQUIRE(relu(0) == 0);
  REQUIRE(relu(200) == 200);
  REQUIRE(relu(99999999999999) == 99999999999999);
}

TEST_CASE("Sigmoid returns the right answers", "[function]")
{
  REQUIRE(sigmoid(0) == 0.5);
  CHECK(sigmoid(1) == 0.7310585786);
  CHECK(sigmoid(10) == 0.9999546);
  CHECK(sigmoid(-1) == 0.2689414);
  CHECK(sigmoid(-10) == 0.0000454);
}

TEST_CASE("Sigmoid has the right boundaries", "[function]")
{
  REQUIRE(sigmoid(17) <= 1);
  REQUIRE(sigmoid(-17) >= 0);
}