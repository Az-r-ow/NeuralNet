#pragma once

#include <Eigen/Dense>

namespace NeuralNet
{
  class Optimizer
  {
  public:
    Optimizer(double alpha) : alpha(alpha){};

    virtual void updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &weightsGrad) const = 0;
    virtual void updateBiases(Eigen::MatrixXd &biases, const Eigen::MatrixXd &biasesGrad) const = 0;

    virtual ~Optimizer() = default; // Virtual destructor
  protected:
    double alpha;
  };
}