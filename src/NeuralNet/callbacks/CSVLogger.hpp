#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "Callback.hpp"
#include "utils/Functions.hpp" // fileExistsWithExtension

namespace NeuralNet
{
  class CSVLogger : public Callback
  {
  public:
    CSVLogger(const std::string &filename, const std::string &separator = ",")
    {
      assert(fileExistsWithExtension(filename, ".csv") && "The file doesn't exists or is not a CSV file '.csv'");
      this->filename = filename;
      this->separator = separator;
    };

    void onEpochBegin(Logs logs) override{};

    void onEpochEnd(Logs logs) override
    {
      std::vector<double> row;

      row.reserve(logs.size());

      std::transform(logs.begin(), logs.end(), std::back_inserter(row),
                     [](const auto &log)
                     { return static_cast<double>(log.second); });

      data.push_back(row);
    };

    void onTrainBegin(Logs logs) override
    {
      // Initializing the headers with the logs keys
      for (const auto &log : logs)
      {
        headers.push_back(log.first);
      };
    };

    void onTrainEnd(Logs logs) override
    {
      std::ofstream file(filename);

      if (!file.is_open())
      {
        throw std::runtime_error("Couldn't open csv file");
      }

      file << formatRow(headers);

      std::for_each(data.begin(), data.end(), [&file, this](auto &row)
                    { file << this->formatRow(row); });

      file.close();
    };

    void onBatchBegin(Logs logs) override{};
    void onBatchEnd(Logs logs) override{};

  private:
    std::string filename;
    std::string separator;
    std::vector<std::string> headers;
    std::vector<std::vector<double>> data;

    template <typename T>
    std::string formatRow(const std::vector<T> &v)
    {
      std::string csvRow;

      for (T el : v)
      {
        csvRow += std::to_string(el) + separator;
      }

      return csvRow + "\n";
    };

    std::string formatRow(const std::vector<std::string> &v)
    {
      std::string csvRow;

      for (const std::string &el : v)
      {
        csvRow += el + separator;
      }

      return csvRow + "\n";
    };
  };
} // namespace NeuralNet