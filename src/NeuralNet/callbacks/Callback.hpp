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
    virtual bool onTrainBegin(Logs logs) = 0;
    virtual bool onTrainEnd(Logs logs) = 0;
    virtual bool onEpochBegin(Logs logs) = 0;
    virtual bool onEpochEnd(Logs logs) = 0;
    virtual bool onBatchBegin(Logs logs) = 0;
    virtual bool onBatchEnd(Logs logs) = 0;

    virtual ~Callback() = default;

    template <typename T>
    using MethodPointer = bool (T::*)(Logs logs);

    template <typename T, typename... Args>
    static bool callMethod(std::shared_ptr<T> callback, const std::string &methodName, Logs logs)
    {
      static const std::unordered_map<std::string, std::function<bool(T *, Logs)>> methods = {
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

      if (it != methods.end())
        return it->second(callback.get(), logs);

      return true;
    }

  protected:
    static void checkMetric(const std::string &metric, const std::vector<std::string> &metrics)
    {
      if (std::find(metrics.begin(), metrics.end(), metric) == metrics.end())
        throw std::invalid_argument("Metric not found");
    };
  };
}