#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Callback.hpp"
#include "utils/Functions.hpp"  // fileExistsWithExtension

namespace NeuralNet {
class CSVLogger : public Callback {
 public:
  /**
   * @brief CSVLogger is a `Callback` that streams epoch results to a csv file
   *
   * @param filepath The name of the csv file
   * @param separator The separator used in the csv file (default: ",")
   */
  CSVLogger(const std::string &filepath, const std::string &separator = ",") {
    assert(fileHasExtension(filepath, ".csv") &&
           "filepath must have .csv extension");
    this->filepath = filepath;
    this->separator = separator;
  };

  void onEpochBegin(Model &model) override {};

  /**
   * @brief This method will be called at the end of each epoch
   *
   * In the case of CSVLogger, it will append the logs of the current epoch to
   * data which `onTrainEnd` will be written to the file.
   *
   * @param logs The logs of the current epoch
   */
  void onEpochEnd(Model &model) override {
    std::unordered_map<std::string, Logs> logs = getLogs(model);
    std::vector<double> row;

    row.reserve(logs.size());

    std::transform(logs.begin(), logs.end(), std::back_inserter(row),
                   [](const auto &log) {
                     const auto &value = log.second;
                     if (std::holds_alternative<int>(value)) {
                       return static_cast<double>(std::get<int>(value));
                     }
                     return std::get<double>(value);
                   });

    data.push_back(row);
  };

  /**
   * @brief This method will be called at the beginning of the training.
   *
   * It will initialize the headers with the logs keys.
   *
   * @param logs The logs of the current epoch
   */
  void onTrainBegin(Model &model) override {
    std::unordered_map<std::string, Logs> logs = getLogs(model);
    // Initializing the headers with the logs keys
    for (const auto &log : logs) {
      const auto &value = log.second;
      if (std::holds_alternative<int>(value) ||
          std::holds_alternative<double>(value)) {
        headers.push_back(log.first);
      }
    };
  };

  /**
   * @brief This method will be called at the end of the training.
   *
   * It will write the data in the given csv file.
   *
   * @param logs The logs of the current epoch
   */
  void onTrainEnd(Model &model) override {
    std::ofstream file(filepath);

    if (!file.is_open()) {
      throw std::runtime_error("Couldn't open csv file");
    }

    file << formatRow(headers);

    std::for_each(data.begin(), data.end(),
                  [&file, this](auto &row) { file << this->formatRow(row); });

    file.close();
  };

  void onBatchBegin(Model &model) override {};
  void onBatchEnd(Model &model) override {};

 private:
  std::string filepath;
  std::string separator;
  std::vector<std::string> headers;
  std::vector<std::vector<double>> data;

  /**
   * @brief This method will format a row of the csv file
   *
   * @tparam T The type of the elements in the row
   * @param v The row to format
   *
   * @return The row in a csv format
   */
  template <typename T>
  std::string formatRow(const std::vector<T> &v) {
    std::string csvRow;

    for (T el : v) {
      csvRow += std::to_string(el) + separator;
    }

    // Remove last ","
    csvRow.pop_back();

    return csvRow + "\n";
  };

  std::string formatRow(const std::vector<std::string> &v) {
    std::string csvRow;

    for (const std::string &el : v) {
      csvRow += el + separator;
    }

    // Remove last ","
    csvRow.pop_back();

    return csvRow + "\n";
  };
};
}  // namespace NeuralNet
