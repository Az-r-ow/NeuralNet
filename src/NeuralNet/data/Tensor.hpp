#pragma once

#include <vector>

namespace NeuralNet
{
  template <typename T>
  class Tensor
  {
    friend class Network;

  public:
    Tensor(T data) : data(data) {}

    void batch(int batchSize)
    {
      assert(data.size() > 0 && batchSize < data.size());

      int numBatches = (data.size() + batchSize - 1) / batchSize;

      batches.reserve(numBatches);

      batched = true;

      for (int i = 0; i < numBatches; ++i)
      {
        int startIdx = i * batchSize;
        int endIdx = std::min((i + 1) * batchSize, static_cast<int>(data.size()));

        batches.emplace_back(std::make_move_iterator(data.begin() + startIdx), std::make_move_iterator(data.begin() + endIdx));
      }

      // Remove the the mess from data
      data.erase(data.begin(), data.end());
    };

    size_t size() const
    {
      return batched ? data.size() : batches.size();
    };

    std::vector<T> getBatchedData() const
    {
      return batches;
    }

    std::vector<T> getData() const
    {
      return data;
    }

  private:
    T data;
    std::vector<T> batches;
    bool batched = false;
  };
}