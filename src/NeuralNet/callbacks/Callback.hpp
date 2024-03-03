#pragma once

#include <vector>
#include <variant>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>

namespace NeuralNet
{
  using Logs = std::unordered_map<std::string, double>;

  class Callback
  {
  public:
    virtual void onTrainBegin(Logs logs) = 0;
    virtual void onTrainEnd(Logs logs) = 0;
    virtual void onEpochBegin(Logs logs) = 0;
    virtual void onEpochEnd(Logs logs) = 0;
    virtual void onBatchBegin(Logs logs) = 0;
    virtual void onBatchEnd(Logs logs) = 0;

    virtual ~Callback() = default;

    /**
     * @brief Calls the method of the callback with the given logs
     *
     * @tparam T The type of the callback
     * @param callback A shared_ptr to the callback
     * @param methodName The name of the method to call (onTrainBegin, onTrainEnd, onEpochBegin, onEpochEnd, onBatchBegin, onBatchEnd)
     * @param logs The logs to pass to the method
     *
     * @warning There should be consistency between the names of the logs and the metrics of the callbacks
     */
    template <typename T>
    static void callMethod(std::shared_ptr<T> callback, const std::string &methodName, Logs logs)
    {
      static const std::unordered_map<std::string, std::function<void(T *, Logs)>> methods = {
          {"onTrainBegin", [](T *callback, Logs logs)
           { return callback->onTrainBegin(logs); }},
          {"onTrainEnd", [](T *callback, Logs logs)
           { return callback->onTrainEnd(logs); }},
          {"onEpochBegin", [](T *callback, Logs logs)
           { return callback->onEpochBegin(logs); }},
          {"onEpochEnd", [](T *callback, Logs logs)
           { return callback->onEpochEnd(logs); }},
          {"onBatchBegin", [](T *callback, Logs logs)
           { return callback->onBatchBegin(logs); }},
          {"onBatchEnd", [](T *callback, Logs logs)
           { return callback->onBatchEnd(logs); }}};

      auto it = methods.find(methodName);

      if (it == methods.end())
        return;

      it->second(callback.get(), logs);
    }

  protected:
    static void checkMetric(const std::string &metric, const std::vector<std::string> &metrics)
    {
      if (std::find(metrics.begin(), metrics.end(), metric) == metrics.end())
        throw std::invalid_argument("Metric not found");
    };
  };
} // namespace NeuralNet