#include <catch2/catch_test_macros.hpp>
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

TEST_CASE("Sqr function returns the right square", "[function]")
{
  CHECK(sqr(0) == 0);
  CHECK(sqr(2) == 4);
  CHECK(sqr(10) == 100);
  CHECK(sqr(7.8877) == 62.21581129);
  CHECK(sqr(657666) == 432524567556);
}