#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <utils/Functions.hpp>

using namespace Catch::Matchers;
using namespace NeuralNet;

const double ERR_MARGIN = 0.001;

TEST_CASE("Sqr function returns the right square", "[function]")
{
  CHECK(sqr(0) == 0);
  CHECK(sqr(2) == 4);
  CHECK(sqr(10) == 100);
  REQUIRE_THAT(sqr(7.8877), WithinAbs(62.215, ERR_MARGIN));
  CHECK(sqr(657666) == 432524567556);
}