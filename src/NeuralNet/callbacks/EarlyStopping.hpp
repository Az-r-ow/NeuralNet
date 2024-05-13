#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "Callback.hpp"
#include "utils/Functions.hpp"

namespace NeuralNet {
class EarlyStopping : public Callback {
 public:
  /**
   * @brief EarlyStopping is a `Callback` that stops training when a monitored
   * metric has stopped improving.
   *
   * @param metric The metric to monitor (default: `LOSS`)
   * @param minDelta Minimum change in the monitored quantity to qualify as an
   * improvement, i.e. an absolute change of less than minDelta, will count as
   * no improvement. (default: 0)
   * @param patience Number of epochs with no improvement after which training
   * will be stopped. (default: 0)
   */
  EarlyStopping(const std::string& metric = "LOSS", double minDelta = 0,
                int patience = 0) {
    checkMetric(metric, metrics);
    this->metric = metric;
    this->minDelta = minDelta;
    this->patience = patience;
  };

  void onEpochBegin(Model& model) override {};

  /**
   * @brief This method will be called at the end of each epoch
   *
   * @param epoch The current epoch
   * @param logs The logs of the current epoch
   * @return Returns true if the training should continue otherwise returns
   * false
   *
   * @warning The order of the logs should be the same as the order of the
   * metrics.
   */
  void onEpochEnd(Model& model) override {
    std::unordered_map<std::string, Logs> logs = getLogs(model);
    auto it = logs.find(metric);

    if (it == logs.end()) throw std::invalid_argument("Metric not found");

    double currentMetric = std::get<double>(it->second);

    if (previousMetric == 0) {
      previousMetric = currentMetric;
      return;
    }

    double absCurrentDelta = std::abs(currentMetric - previousMetric);

    patience = absCurrentDelta <= minDelta ? patience - 1 : patience;
    previousMetric = currentMetric;

    if (patience < 0) throw std::runtime_error("Early stopping");
  };

  void onTrainBegin(Model& model) override {};
  void onTrainEnd(Model& model) override {};
  void onBatchBegin(Model& model) override {};
  void onBatchEnd(Model& model) override {};

  ~EarlyStopping() override = default;

 private:
  std::string metric;
  double minDelta, previousMetric = 0;
  int patience;
  std::vector<std::string> metrics = {
      "LOSS", "ACCURACY"};  // Available metrics for this Callback
};

}  // namespace NeuralNet