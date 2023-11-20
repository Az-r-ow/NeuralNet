#pragma once

#include "Tensor.hpp"

namespace NeuralNet
{
  template <typename I, typename L>
  class TrainingData
  {
    friend class Network;

  public:
    TrainingData(I inputs_data, L labels_data) : inputs(inputs_data), labels(labels_data)
    {
      assert(inputs_data.size() == labels_data.size());
    }

    void batch(int batchSize)
    {
      inputs.batch(batchSize);
      labels.batch(batchSize);
      batched = true;
    };

  private:
    Tensor<I> inputs;
    Tensor<L> labels;
    bool batched = false;
  };
}