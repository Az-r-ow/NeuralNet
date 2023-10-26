#pragma once

#include "interfaces/Optimizer.hpp"

namespace NeuralNet
{
  /**
   * Stochastic Gradient Descent optimizer
   */
  class SGD : public Optimizer
  {
  public:
    SGD(double alpha) : Optimizer(alpha){};

    ~SGD() override = default;

  private:
    void updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &weightsGrad) override
    {
      weights = weights.array() - (this->alpha * weightsGrad).array();
    };

    void updateBiases(Eigen::MatrixXd &biases, const Eigen::MatrixXd &biasesGrad) override
    {
      biases = biases.array() - (this->alpha * biasesGrad).array();
    };

    void insiderInit(size_t size) override{};
  };
}