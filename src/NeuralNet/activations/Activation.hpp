#pragma once

#include <Eigen/Dense>

namespace NeuralNet
{
  class Activation
  {
  public:
    /**
     * @brief Activate a layer's outputs
     *
     * @param z A matrix representing a layer's outputs
     *
     * @return The activated outputs
     */
    static Eigen::MatrixXd activate(const Eigen::MatrixXd &z);

    /**
     * @brief Compute the derivative of the activation function
     *
     * @param a Activated outputs
     *
     * @return The activated outputs derivatives
     */
    static Eigen::MatrixXd diff(const Eigen::MatrixXd &a);
  };
}
