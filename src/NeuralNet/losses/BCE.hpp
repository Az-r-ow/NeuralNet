#pragma once

#include "Loss.hpp"

namespace NeuralNet {
/**
 * Binary Cross-Entropy
 */
class BCE : public Loss {
 public:
  static double cmpLoss(const Eigen::MatrixXd &o, const Eigen::MatrixXd &y) {
    Eigen::MatrixXd loss =
        -(y.array() * o.array().log() + (1.0 - y.array()).log());
    return loss.sum();
  }

  static Eigen::MatrixXd cmpLossGrad(const Eigen::MatrixXd &yHat,
                                     const Eigen::MatrixXd &y) {
    return (yHat.array() - y.array()) / (yHat.array() * (1.0 - y.array()));
  }
};

}  // namespace NeuralNet