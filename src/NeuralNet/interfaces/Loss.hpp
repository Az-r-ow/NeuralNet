#pragma once

#include <Eigen/Dense>
#include "utils/Typedefs.hpp"

namespace NeuralNet
{
  class Loss
  {
  public:
    static double cmpLoss(const Eigen::MatrixXd &o, const Labels &y);
    static Eigen::MatrixXd cmpGradient(const Eigen::MatrixXd &o, const Labels &y);
  };
}