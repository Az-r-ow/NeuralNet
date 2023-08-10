#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Functions.hpp>

using namespace NeuralNet;
using namespace Catch::Matchers;

const double ERR_MARGIN = 0.001;

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
  REQUIRE_THAT(sigmoid(1), WithinAbs(0.731, ERR_MARGIN));
  REQUIRE_THAT(sigmoid(10), WithinAbs(1, ERR_MARGIN));
  REQUIRE_THAT(sigmoid(-1), WithinAbs(0.268, ERR_MARGIN));
  REQUIRE_THAT(sigmoid(-10), WithinAbs(0, ERR_MARGIN));
}

TEST_CASE("Sqr function returns the right square", "[function]")
{
  CHECK(sqr(0) == 0);
  CHECK(sqr(2) == 4);
  CHECK(sqr(10) == 100);
  REQUIRE_THAT(sqr(7.8877), WithinAbs(62.215, ERR_MARGIN));
  CHECK(sqr(657666) == 432524567556);
}