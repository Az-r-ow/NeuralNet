#pragma once

#include "Activation.hpp"

namespace NeuralNet {
class Relu : public Activation {
 public:
  static Eigen::MatrixXd activate(const Eigen::MatrixXd &z) {
    return z.unaryExpr(&Relu::activateValue);
  }

  static Eigen::MatrixXd diff(const Eigen::MatrixXd &a) {
    return a.unaryExpr(&Relu::diffValue);
  }

 private:
  static double diffValue(double a) { return a > 0 ? 1 : 0; }

  static double activateValue(double z) { return z < 0 ? 0 : z; }
};
}  // namespace NeuralNet