#include <Network.hpp>
#include <callbacks/Callback.hpp>
#include <callbacks/EarlyStopping.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <utils/Variants.hpp>
#include <vector>

using namespace NeuralNet;

TEST_CASE(
    "EarlyStopping callback throws exception when the metric is not found",
    "[callback]") {
  std::shared_ptr<Callback> earlyStopping =
      std::make_shared<EarlyStopping>("NOT_A_METRIC", 0.1);

  Network network;

  REQUIRE_THROWS(Callback::callMethod(earlyStopping, "onEpochEnd", network));
}

TEST_CASE(
    "EarlyStopping callback throws exception when metric does not vary more "
    "than the given delta",
    "[callback]") {
  std::shared_ptr<Callback> earlyStopping =
      std::make_shared<EarlyStopping>("LOSS", 0.1);

  Network network;

  std::shared_ptr<Optimizer> optimizer = std::make_shared<SGD>(1);

  network.setup(optimizer, LOSS::QUADRATIC);

  std::shared_ptr<Layer> inputLayer = std::make_shared<Dense>(2);
  std::shared_ptr<Layer> outputLayer = std::make_shared<Dense>(1);

  network.addLayer(inputLayer);
  network.addLayer(outputLayer);

  std::vector<std::vector<double>> inputs = {{0, 0}};

  network.train(inputs, {0}, 1, {}, false);

  Callback::callMethod(earlyStopping, "onEpochEnd", network);
  CHECK_THROWS(Callback::callMethod(earlyStopping, "onEpochEnd", network));
}