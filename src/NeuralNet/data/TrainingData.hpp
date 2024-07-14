#pragma once

#include <algorithm>
#include <random>
#include <utility>  // For std::pair
#include <vector>

namespace NeuralNet {
template <typename X, typename Y>
class TrainingData {
  friend class Network;

 public:
  /**
   * @brief Construct a new Training Data object.
   * This object is used to store the inputs and labels data,
   * it comes with a set of methods to manipulate the data to your liking for
   * better training optimization.
   *
   * @param inputs_data The inputs data
   * @param labels_data The labels data
   *
   * @note The inputs and labels data must have the same size
   */
  TrainingData(X xTrain, Y yTrain, X xTest = X(), Y yTest = Y())
      : xTrain(xTrain), yTrain(yTrain), xTest(xTest), yTest(yTest) {
    assert(xTrain.size() == yTrain.size());
    if (!xTest.empty() && !yTest.empty()) assert(xTest.size() == xTest.size());
  }

  std::vector<std::pair<X, Y>> getMiniBatches() { return this->miniBatches; }

  /**
   * @brief This method will separate the inputs and labels data into batches of
   * the specified size
   *
   * @param batchSize The number of elements in each batch
   */
  void batch(int batchSize, bool stratified = false, bool shuffle = false,
             bool dropLast = false, bool verbose = false) {
    batched = true;
    if (!stratified)
      return normalMiniBatch(batchSize, xTrain, yTrain, shuffle, dropLast,
                             verbose);
    return stratifiedMiniBatch(batchSize, xTrain, yTrain, shuffle, dropLast,
                               verbose);
  };

 private:
  X xTrain;
  Y yTrain;
  X xTest;
  Y yTest;
  std::vector<std::pair<X, Y>> miniBatches;
  bool batched = false;

  void printDropLast(const size_t size, const size_t requiredSize) {
    std::cout << "Dropping last mini-batch (size : " << size << " < "
              << requiredSize << ")" << std::endl;
  };

  void printNBatchesCreated(const size_t nBatches) {
    std::cout << "Total mini-batches created : " << nBatches << std::endl;
  }

  template <typename Ti, typename Tl>
  void shuffle(std::vector<Ti>& inputs, std::vector<Tl>& labels) {
    // Create a vector of pairs (data, label)
    std::vector<std::pair<Ti, Tl>> combined;
    for (size_t i = 0; i < inputs.size(); ++i) {
      combined.emplace_back(inputs[i], labels[i]);
    }

    // Create a random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffle the combined vector
    std::shuffle(combined.begin(), combined.end(), g);

    // Separate the shuffled data and labels
    for (size_t i = 0; i < combined.size(); ++i) {
      inputs[i] = combined[i].first;
      labels[i] = combined[i].second;
    }
  };

  void normalMiniBatch(int batchSize, X& x, Y& y, bool shuffle = false,
                       bool dropLast = false, bool verbose = false) {
    int nInputs = x.size();
    assert(nInputs > 0 && batchSize < nInputs);
    int nMiniBatches = (nInputs + batchSize - 1) / batchSize;

    if (shuffle) {
      this->shuffle(x, y);
    }

    miniBatches.reserve(nMiniBatches);

    for (int i = 0; i < nMiniBatches; ++i) {
      int startIdx = i * batchSize;
      int endIdx = std::min((i + 1) * batchSize, static_cast<int>(nInputs));

      X xMiniBatch(x.begin() + startIdx, x.begin() + endIdx);
      Y yMiniBatch(y.begin() + startIdx, y.begin() + endIdx);

      std::pair<X, Y> miniBatch = std::make_pair(xMiniBatch, yMiniBatch);

      if (xMiniBatch.size() < static_cast<size_t>(batchSize) && dropLast) {
        if (verbose)
          printDropLast(xMiniBatch.size(), static_cast<size_t>(batchSize));
        continue;
      }

      miniBatches.push_back(miniBatch);
    }

    // empty the respective containers
    x.erase(x.begin(), x.end());
    y.erase(y.begin(), y.end());

    if (verbose) printNBatchesCreated(miniBatches.size());
  };

  template <typename Tx, typename Ty>
  void stratifiedMiniBatch(int batchSize, std::vector<Tx>& x,
                           std::vector<Ty>& y, bool shuffle = false,
                           bool dropLast = false, bool verbose = false) {
    // Group data by class
    std::map<Ty, X> classDataMap;
    for (size_t i = 0; i < x.size(); i++) {
      classDataMap[y[i]].push_back(x[i]);
    }

    // Calculate number of samples per class in each batch
    std::map<Ty, int> classBatchSize;
    int totalSamples = x.size();
    for (const auto& classPair : classDataMap) {
      int classCount = classPair.second.size();
      classBatchSize[classPair.first] = (classCount * batchSize) / totalSamples;

      if (verbose)
        std::cout << "Class count for (" << classPair.first
                  << ") = " << classCount
                  << " - classBatchSize = " << classBatchSize[classPair.first]
                  << std::endl;
    }

    // Create batches
    bool moreData = true;
    while (moreData) {
      X batchData;
      Y batchLabels;
      moreData = false;

      // Fill mini-batch with value of each class
      for (auto& classPair : classDataMap) {
        Ty classLabel = classPair.first;
        X& classSamples = classPair.second;
        int nSamplesToAdd = classBatchSize[classLabel];

        for (int i = 0; i < nSamplesToAdd && !classSamples.empty(); i++) {
          batchData.push_back(classSamples.back());
          batchLabels.push_back(classLabel);
          classSamples.pop_back();
        }

        if (!classSamples.empty()) {
          moreData = true;
        }
      }

      while (batchData.size() < static_cast<size_t>(batchSize) && moreData) {
        bool added = false;
        for (auto& classPair : classDataMap) {
          if (!classPair.second.empty()) {
            batchData.push_back(classPair.second.back());
            batchLabels.push_back(classPair.first);
            classPair.second.pop_back();
            added = true;
            if (batchData.size() == static_cast<size_t>(batchSize)) break;
          }
        }

        if (!added) break;
      }

      if (batchData.size() < static_cast<size_t>(batchSize) && dropLast) {
        // skip placing it in miniBatches
        if (verbose)
          printDropLast(batchData.size(), static_cast<size_t>(batchSize));
        continue;
      }

      if (!batchData.empty()) {
        // Shuffle batches if indicated
        if (shuffle) this->shuffle(batchData, batchLabels);
        miniBatches.emplace_back(batchData, batchLabels);
      }
    }

    if (verbose) printNBatchesCreated(miniBatches.size());
  }
};
}  // namespace NeuralNet