#pragma once

#include "Tensor.hpp"

namespace NeuralNet {
template <typename I, typename L>
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
  TrainingData(I inputs_data, L labels_data)
      : inputs(inputs_data), labels(labels_data) {
    assert(inputs_data.size() == labels_data.size());
  }

  /**
   * @brief This method will separate the inputs and labels data into batches of
   * the specified size
   *
   * @param batchSize The number of elements in each batch
   */
  void batch(int batchSize) {
    inputs.batch(batchSize);
    labels.batch(batchSize);
    batched = true;
  };

 private:
  Tensor<I> inputs;
  Tensor<L> labels;
  bool batched = false;
};
}  // namespace NeuralNet