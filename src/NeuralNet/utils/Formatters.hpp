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

  static Labels formatLabels(vector<double> y, int size)
  {
    assert(y.size() == size);
    return MatrixXd::Map(&y[0], size, 1);
  };
}