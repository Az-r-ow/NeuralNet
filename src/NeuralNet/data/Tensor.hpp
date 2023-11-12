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
      // For now only 3 dimensional tensors can be managed
      assert(vDepth(data) > 3 && "Data format not accepted");

      this->data = data;
    };

    void batch(int batchSize);

  private:
    // Each element in data will be considered a batch
    // Batching would be encapsulating the elements in an arrays
    std::vector<T> data;

    int vDepth(std::vector<T> v)
    {
      if (v.size() == 0 || !std::is_vector < decltype(v[0])::value)
      {
        return 1;
      }

      int nestedDepth = vDepth(v[0]);

      return 1 + nestedDepth;
    }
  };
}