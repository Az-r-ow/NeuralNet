#pragma once

#include "interfaces/Optimizer.hpp"

namespace NeuralNet
{
  class SGD : public Optimizer
  {
  public:
    SGD(double alpha) : Optimizer(alpha){};

    void updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &weightsGrad) override
    {
      weights = weights.array() - (this->alpha * weightsGrad.transpose()).array();
    };

    void updateBiases(Eigen::MatrixXd &biases, const Eigen::MatrixXd &biasesGrad) override
    {
      biases = biases.array() - (this->alpha * biasesGrad.transpose()).array();
    };

    ~SGD() override = default;
  };
}