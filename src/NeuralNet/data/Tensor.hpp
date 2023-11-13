#pragma once

#include <vector>

namespace NeuralNet
{
  template <typename T>
  class Tensor
  {
  public:
    Tensor(std::vector<T> data)
    {
      this->data = data;
    };

    void batch(int batchSize)
    {
      assert(data.size() > 0 && batchSize < data.size());

      int numBatches = (data.size() + batchSize - 1) / batchSize;

      batches.reserve(numBatches);

      for (int i = 0; i < numBatches; ++i)
      {
        int startIdx = i * batchSize;
        int endIdx = std::min((i + 1) * batchSize, static_cast<int>(data.size()));

        batches.emplace_back(data.begin() + startIdx, data.begin() + endIdx);
      }
    };

    std::vector<std::vector<T>> getBatchedData() const
    {
      return batches;
    }

  private:
    std::vector<T> data;
    std::vector<std::vector<T>> batches;
  };
}