#pragma once

#include <Eigen/Core>
#include <type_traits>
#include "Functions.hpp"

namespace NeuralNet
{
  /**
   * @brief This function transform the labels in 2d or 1d vectors into Matrices
   *
   * The resulting matrices will be used to evaluate the model's performance and do the necessary adjustments on its parameters
   *
   * @param labels 1d std::vector or 2d std::vector (based on the task)
   * @param shape std::tuple that defines the shape of the resulting matrix
   *
   * @return The resulting matrix from the passed labels
   */
  template <typename T>
  static Eigen::MatrixXd formatLabels(std::vector<T> labels, std::tuple<int, int> shape)
  {
    int rows = std::get<0>(shape);
    int cols = std::get<1>(shape);

    // When propagating forward, the number of rows will be the number of inputs
    assert(labels.size() == rows && "The number of labels don't match the number of inputs");

    Eigen::MatrixXd mLabels(rows, cols);

    if constexpr (std::is_same<T, std::vector<double>>::value)
    {
      std::vector<double> flattenedVector = flatten2DVector(labels, rows, cols);
      mLabels = Eigen::Map<Eigen::MatrixXd>(flattenedVector.data(), rows, cols);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
      mLabels = Eigen::MatrixXd::Zero(rows, cols);

      /**
       * Setting the cols indexes to 1
       * Notably used in classification tasks
       */
      for (int i = 0; i < rows; i++)
      {
        int colIndex = labels[i];
        assert(colIndex < cols);
        mLabels(i, colIndex) = 1;
      }
    }

    return mLabels;
  }
} // namespace NeuralNet
