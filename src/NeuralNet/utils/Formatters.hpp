#pragma once

#include <Eigen/Core>
#include "Typedefs.hpp"
#include "Functions.hpp"

namespace NeuralNet
{
  /**
   * @brief This function takes an integer and the size of the outputs and generates labels from integers
   *
   * @param y the label
   * @param size the size (number of labels)
   *
   * @return a vector of Labels
   */
  static Labels formatLabels(int y, int size)
  {
    assert(y >= 0 && y < size);
    Labels labels = Eigen::MatrixXd::Zero(size, 1);
    labels(y) = 1;
    return labels;
  };

  /**
   * @brief This function takes a std::vector as input and transforms it into an Eigen::Vector
   *
   * @param y std::vector
   * @param size the size of the desired vector
   *
   * @return a vector of Labels
   */
  static Labels formatLabels(std::vector<double> y, int size)
  {
    assert(y.size() == size);
    return Eigen::MatrixXd::Map(&y[0], size, 1);
  };

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

    if (std::is_same<T, std::vector<double>>)
    {
      std::vector<double> flattenedVector = flatten2DVector(labels, rows, cols);
      mLabels = Eigen::Map<Eigen::MatrixXd>(flattenedVector.data(), rows, cols);
    }
    else if (std::is_same<T, double>)
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

  /**
   * @brief Converts layer's outputs from Eigen::Eigen::MatrixXd to an std::vector
   *
   * @param outputs the model's outputs
   *
   * @return outputs as an std::vector
   */
  static std::vector<double> formatOutputs(Eigen::MatrixXd outputs)
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