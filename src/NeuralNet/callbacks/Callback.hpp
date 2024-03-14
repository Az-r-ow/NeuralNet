#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace NeuralNet {

class Callback {
 public:
  virtual void onTrainBegin(std::unordered_map<std::string, double> logs) = 0;
  virtual void onTrainEnd(std::unordered_map<std::string, double> logs) = 0;
  virtual void onEpochBegin(std::unordered_map<std::string, double> logs) = 0;
  virtual void onEpochEnd(std::unordered_map<std::string, double> logs) = 0;
  virtual void onBatchBegin(std::unordered_map<std::string, double> logs) = 0;
  virtual void onBatchEnd(std::unordered_map<std::string, double> logs) = 0;

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
                         const std::string &methodName,
                         std::unordered_map<std::string, double> logs) {
    static const std::unordered_map<
        std::string,
        std::function<void(T *, std::unordered_map<std::string, double>)>>
        methods = {
            {"onTrainBegin",
             [](T *callback, std::unordered_map<std::string, double> logs) {
               return callback->onTrainBegin(logs);
             }},
            {"onTrainEnd",
             [](T *callback, std::unordered_map<std::string, double> logs) {
               return callback->onTrainEnd(logs);
             }},
            {"onEpochBegin",
             [](T *callback, std::unordered_map<std::string, double> logs) {
               return callback->onEpochBegin(logs);
             }},
            {"onEpochEnd",
             [](T *callback, std::unordered_map<std::string, double> logs) {
               return callback->onEpochEnd(logs);
             }},
            {"onBatchBegin",
             [](T *callback, std::unordered_map<std::string, double> logs) {
               return callback->onBatchBegin(logs);
             }},
            {"onBatchEnd",
             [](T *callback, std::unordered_map<std::string, double> logs) {
               return callback->onBatchEnd(logs);
             }}};

    auto it = methods.find(methodName);

    if (it == methods.end()) return;

    it->second(callback.get(), logs);
  }

 protected:
  static void checkMetric(const std::string &metric,
                          const std::vector<std::string> &metrics) {
    if (std::find(metrics.begin(), metrics.end(), metric) == metrics.end())
      throw std::invalid_argument("Metric not found");
  };
};
}  // namespace NeuralNet