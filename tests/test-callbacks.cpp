#include <callbacks/Callback.hpp>
#include <callbacks/EarlyStopping.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace NeuralNet;

TEST_CASE(
    "EarlyStopping callback throws exception when the metric is not found",
    "[callback]") {
  std::shared_ptr<Callback> earlyStopping =
      std::make_shared<EarlyStopping>("LOSS", 0.1);
  std::unordered_map<std::string, double> logs = {{"TEST", 0.2}};

  REQUIRE_THROWS(Callback::callMethod(earlyStopping, "onEpochEnd", logs));
}

TEST_CASE(
    "EarlyStopping callback throws exception when metric does not more than "
    "the given delta",
    "[callback]") {
  std::shared_ptr<Callback> earlyStopping =
      std::make_shared<EarlyStopping>("LOSS", 0.1);
  std::unordered_map<std::string, double> logs = {{"LOSS", 0.2}};
  std::unordered_map<std::string, double> logs2 = {{"LOSS", 0.2}};

  Callback::callMethod(earlyStopping, "onEpochEnd", logs);
  REQUIRE_THROWS(Callback::callMethod(earlyStopping, "onEpochEnd", logs2));
}