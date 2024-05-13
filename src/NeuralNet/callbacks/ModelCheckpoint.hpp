#pragma once

#include <format>
#include <memory>
#include <type_traits>

#include "Callback.hpp"
#include "Model.hpp"
#include "utils/Functions.hpp"

namespace NeuralNet {
class ModelCheckpoint : public Callback {
 public:
  ModelCheckpoint(const std::string &folderPath, const bool saveBestOnly = true,
                  const int numEpochs = 1, const bool verbose = false) {
    assert(folderExists(folderPath) && "Folder doesn't exist");
    this->folderPath = folderPath;
    this->saveBestOnly = saveBestOnly;
    this->numEpochs = numEpochs;
    this->verbose = verbose;
  };

  void onEpochBegin(Model &model) override {};

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
  void onEpochEnd(Model &model) override {
    std::unordered_map<std::string, Logs> logs = getLogs(model);
    int epoch = std::get<int>(logs.at("EPOCH"));
    std::string filename = "checkpoint-" + std::to_string(epoch) + ".bin";

    if (saveBestOnly) {
      double currLoss = std::get<double>(logs.at("LOSS"));
      double currAccuracy = std::get<double>(logs.at("ACCURACY"));

      if (currLoss > bestLoss && bestAccuracy > currAccuracy) return;

      // Save best model for later saving
      bestLoss = currLoss;
      bestAccuracy = currAccuracy;
      bestModel = &model;
      this->filename = filename;  // for later use onTrainEnd
      return;
    }

    if ((epoch % numEpochs) != 0) return;

    if (verbose) verboseOutput(filename);

    model.to_file(filename);
  };

  void onTrainBegin(Model &model) override {};

  void onTrainEnd(Model &model) override {
    if (!saveBestOnly) return;
    if (verbose) verboseOutput(filename);
    bestModel->to_file(filename);
  };

  void onBatchBegin(Model &model) override {};
  void onBatchEnd(Model &model) override {};

  ~ModelCheckpoint() override = default;

 private:
  std::string folderPath, filename;
  bool saveBestOnly, verbose;
  double bestLoss = 10, bestAccuracy = 0;
  Model *bestModel = nullptr;
  int numEpochs, bestEpoch;

  void verboseOutput(const std::string filename) {
    std::cout << "Saving model in file: " << filename << "\n";
  }
};
}  // namespace NeuralNet