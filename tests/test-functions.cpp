#include <catch2/catch_test_macros.hpp>

TEST_CASE("Relu of negative numbers always returns 0", "[function]")
{
  REQUIRE(relu(-1) == 0);
}