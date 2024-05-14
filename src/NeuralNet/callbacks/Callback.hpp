#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Model.hpp"
#include "utils/Variants.hpp"

namespace NeuralNet {

class Model;

class Callback {
 public:
  virtual void onTrainBegin(Model &model) = 0;
  virtual void onTrainEnd(Model &model) = 0;
  virtual void onEpochBegin(Model &model) = 0;
  virtual void onEpochEnd(Model &model) = 0;
  virtual void onBatchBegin(Model &model) = 0;
  virtual void onBatchEnd(Model &model) = 0;

  virtual ~Callback() = default;

  /**
   * @brief Calls the method of the callback with the given logs
   *
   * @tparam T The type of the callback
   * @param callback A shared_ptr to the callback
   * @param methodName The name of the method to call (onTrainBegin, onTrainEnd,
   * onEpochBegin, onEpochEnd, onBatchBegin, onBatchEnd)
   * @param logs The logs to pass to the method
   *
   * @warning There should be consistency between the names of the logs and the
   * metrics of the callbacks
   */
  template <typename T>
  static void callMethod(std::shared_ptr<T> callback,
                         const std::string &methodName, Model &model) {
    static const std::unordered_map<std::string,
                                    std::function<void(T *, Model &)>>
        methods = {{"onTrainBegin",
                    [](T *callback, Model &model) {
                      return callback->onTrainBegin(model);
                    }},
                   {"onTrainEnd",
                    [](T *callback, Model &model) {
                      return callback->onTrainEnd(model);
                    }},
                   {"onEpochBegin",
                    [](T *callback, Model &model) {
                      return callback->onEpochBegin(model);
                    }},
                   {"onEpochEnd",
                    [](T *callback, Model &model) {
                      return callback->onEpochEnd(model);
                    }},
                   {"onBatchBegin",
                    [](T *callback, Model &model) {
                      return callback->onBatchBegin(model);
                    }},
                   {"onBatchEnd", [](T *callback, Model &model) {
                      return callback->onBatchEnd(model);
                    }}};

    auto it = methods.find(methodName);

    if (it == methods.end()) return;

    it->second(callback.get(), model);
  }

 protected:
  static void checkMetric(const std::string &metric,
                          const std::vector<std::string> &metrics) {
    if (std::find(metrics.begin(), metrics.end(), metric) == metrics.end())
      throw std::invalid_argument("Metric not found");
  };

  static std::unordered_map<std::string, Logs> getLogs(Model &model) {
    std::unordered_map<std::string, Logs> logs;

    logs["EPOCH"] = model.cEpoch;
    logs["ACCURACY"] = model.accuracy;
    logs["LOSS"] = model.loss;

    return logs;
  };
};
}  // namespace NeuralNet