#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "Callback.hpp"
#include "utils/Functions.hpp"

namespace NeuralNet
{
  class EarlyStopping : public Callback
  {
  public:
    /**
     * @brief EarlyStopping is a `Callback` that stops training when a monitored metric has stopped improving.
     *
     * @param metric The metric to monitor default is `LOSS`
     * @param minDelta Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than minDelta, will count as no improvement.
     * @param patience Number of epochs with no improvement after which training will be stopped.
     */
    EarlyStopping(const std::string &metric, double minDelta = 0, int patience = 0)
    {
      checkMetric(metric, metrics);
      this->metric = metric;
      this->minDelta = minDelta;
      this->patience = patience;
    };

    /**
     * @brief This method will be called at the beginning of each epoch
     *
     * @param epoch The current epoch
     * @param logs The logs of the current epoch
     * @return Returns true if the training should continue otherwise returns false
     *
     * @warning The order of the logs should be the same as the order of the metrics.
     */
    bool onEpochBegin(Logs logs) override { return true; };

    /**
     * @brief This method will be called at the end of each epoch
     *
     * @param epoch The current epoch
     * @param logs The logs of the current epoch
     * @return Returns true if the training should continue otherwise returns false
     *
     * @warning The order of the logs should be the same as the order of the metrics.
     */
    bool onEpochEnd(Logs logs) override
    {
      auto it = logs.find(metric);

      if (it == logs.end())
        throw std::invalid_argument("Metric not found");

      double currentMetric = it->second;

      if (previousMetric == 0)
      {
        previousMetric = currentMetric;
        return true;
      }

      double absCurrentDelta = std::abs(currentMetric - previousMetric);

      patience = absCurrentDelta <= minDelta ? patience - 1 : patience;
      previousMetric = currentMetric;

      if (patience < 0)
        return false;

      return true;
    };

    bool onTrainBegin(Logs logs) override { return true; };
    bool onTrainEnd(Logs logs) override { return true; };
    bool onBatchBegin(Logs logs) override { return true; };
    bool onBatchEnd(Logs logs) override { return true; };

    ~EarlyStopping() override = default;

  private:
    std::string metric;
    double minDelta, previousMetric = 0;
    int patience;
    std::vector<std::string> metrics = {"LOSS", "ACCURACY"}; // Available metrics for this Callback
  };

} // namespace NeuralNet