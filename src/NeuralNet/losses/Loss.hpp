#pragma once

#include <Eigen/Dense>

namespace NeuralNet {
class Loss {
 public:
  /**
   * @brief This function computes the loss of the current iteration
   *
   * @param o The outputs from the output layer
   * @param y The labels (the expected values)
   *
   * @return The loss based on the selected loss function
   */
  static double cmpLoss(const Eigen::MatrixXd &o, const Eigen::MatrixXd &y);

  /**
   * @brief This function computes the loss gradient w.r.t the outputs
   *
   * @param o The outputs from the output layer
   * @param y The labels (expected vals)
   *
   * @return The current iteration's gradient
   */
  static Eigen::MatrixXd cmpLossGrad(const Eigen::MatrixXd &o,
                                     const Eigen::MatrixXd &y);
};
}  // namespace NeuralNet