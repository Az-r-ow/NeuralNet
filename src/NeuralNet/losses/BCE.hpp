#pragma once

#include "Loss.hpp"

namespace NeuralNet {
/**
 * Binary Cross-Entropy
 */
class BCE : public Loss {
 public:
  static double cmpLoss(const Eigen::MatrixXd &o, const Eigen::MatrixXd &y) {
    double threshold = 1.0e-5;
    Eigen::MatrixXd oTrim = trim(o, threshold);
    Eigen::MatrixXd yTrim = trim(y, threshold);

    Eigen::MatrixXd loss =
        -(yTrim.array() * oTrim.array().log() +
          (1.0 - yTrim.array()) * (1.0 - oTrim.array()).log());

    if (loss.array().isNaN().any())
      throw std::runtime_error(
          "NaN value encountered. Inputs might be too big");

    return loss.sum();
  }

  static Eigen::MatrixXd cmpLossGrad(const Eigen::MatrixXd &yHat,
                                     const Eigen::MatrixXd &y) {
    return (yHat.array() - y.array()) /
           ((yHat.array() * (1.0 - yHat.array())) + 1e-9);
  }
};

}  // namespace NeuralNet