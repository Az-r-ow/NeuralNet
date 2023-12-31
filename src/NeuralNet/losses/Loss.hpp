#pragma once

#include <Eigen/Dense>
#include "utils/Typedefs.hpp"

namespace NeuralNet
{
  class Loss
  {
  public:
    /**
     * @brief This function computes the loss of the current iteration
     *
     * @param o The outputs from the output layer
     * @param y The labels (the expected values)
     *
     * @return The loss based on the selected loss function
     */
    static double cmpLoss(const Eigen::MatrixXd &o, const Labels &y);

    /**
     * @brief This function computes the gradient w.r.t to the selected loss function
     *
     * @param o The outputs from the output layer
     * @param y The labels (expected vals)
     *
     * @return The current iteration's gradient
     */
    static Eigen::MatrixXd cmpGradient(const Eigen::MatrixXd &o, const Labels &y);
  };
}