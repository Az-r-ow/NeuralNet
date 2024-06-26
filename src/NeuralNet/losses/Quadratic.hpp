#pragma once

#include "Loss.hpp"
#include "utils/Functions.hpp"

namespace NeuralNet {
/**
 * Quadratic Loss
 */
class Quadratic : public Loss {
 public:
  static double cmpLoss(const Eigen::MatrixXd &o, const Eigen::MatrixXd &y) {
    Eigen::MatrixXd cMatrix = o.array() - y.array();
    cMatrix = cMatrix.unaryExpr(&sqr);

    return cMatrix.sum();
  };

  static Eigen::MatrixXd cmpLossGrad(const Eigen::MatrixXd &yHat,
                                     const Eigen::MatrixXd &y) {
    assert(yHat.rows() == y.rows());
    return (yHat.array() - y.array()).matrix() * 2;
  };
};
}  // namespace NeuralNet