#pragma once

#include <Eigen/Core>
#include "Typedefs.hpp"

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