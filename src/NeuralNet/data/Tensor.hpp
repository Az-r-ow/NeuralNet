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

        batches.emplace_back(std::make_move_iterator(data.begin() + startIdx), std::make_move_iterator(data.begin() + endIdx));
      }

      // Remove the the mess from data
      data.erase(data.begin(), data.end());
    };

    std::vector<std::vector<T>> getBatchedData() const
    {
      return batches;
    }

    std::vector<T> getData() const
    {
      return data;
    }

  private:
    std::vector<T> data;
    std::vector<std::vector<T>> batches;
  };
}