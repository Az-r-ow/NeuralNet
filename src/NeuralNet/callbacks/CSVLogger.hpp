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
      assert(fileHasExtension(filename, ".csv") && "Filename must have .csv extension");
      this->filename = filename;
      this->separator = separator;
    };

    void onEpochBegin(Logs logs) override{};

    /**
     * @brief This method will be called at the end of each epoch
     *
     * In the case of CSVLogger, it will append the logs of the current epoch to data which `onTrainEnd` will be written to the file.
     *
     * @param logs The logs of the current epoch
     */
    void onEpochEnd(Logs logs) override
    {
      std::vector<double> row;

      row.reserve(logs.size());

      std::transform(logs.begin(), logs.end(), std::back_inserter(row),
                     [](const auto &log)
                     { return static_cast<double>(log.second); });

      data.push_back(row);
    };

    /**
     * @brief This method will be called at the beginning of the training.
     *
     * It will initialize the headers with the logs keys.
     *
     * @param logs The logs of the current epoch
     */
    void onTrainBegin(Logs logs) override
    {
      // Initializing the headers with the logs keys
      for (const auto &log : logs)
      {
        headers.push_back(log.first);
      };
    };

    /**
     * @brief This method will be called at the end of the training.
     *
     * It will write the data in the given csv file.
     *
     * @param logs The logs of the current epoch
     */
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

    /**
     * @brief This method will format a row of the csv file
     *
     * @tparam T The type of the elements in the row
     * @param v The row to format
     *
     * @return The row in a csv format
     */
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
} // namespa#pragma once

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
} // namespa