#pragma once

#include <Eigen/Core>
#include "Typedefs.hpp"

namespace NeuralNet
{
  static Labels formatLabels(int y, int size)
  {
    Labels labels = MatrixXd::Zero(size, 1);
    labels(y) = 1;
    return labels;
  };

  static Labels formatLabels(std::vector<double> y, int size)
  {
    assert(y.size() == size);
    return MatrixXd::Map(&y[0], size, 1);
  };

  /**
   * Converts layer's outputs to an std::vector
   */
  static std::vector<double> formatOutputs(MatrixXd outputs)
  {
    int i = 0;
    std::vector<double> v;
    v.reserve(outputs.rows()); // Reserving space for efficiency

    while (i < outputs.size())
    {
      v.push_back(outputs(i, 0));
      i++;
    }

    return v;
  }
}