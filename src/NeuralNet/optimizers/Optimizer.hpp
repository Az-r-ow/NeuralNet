#pragma once

#include <Eigen/Dense>

namespace NeuralNet
{
  class Optimizer
  {
    friend class Network;

  public:
    Optimizer(double alpha) : alpha(alpha){};

    virtual ~Optimizer() = default; // Virtual destructor

  protected:
    double alpha;

    /**
     * @brief This function updates the weights passed based on the selected Optimizer and the weights gradients
     *
     * @param weights The weights that should be updated
     * @param weightsGrad The weights gradient
     *
     * The function will return void, since it only performs an update on the weights passed
     */
    virtual void updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &weightsGrad) = 0;

    /**
     * @brief This function updates the biases passed based based on the Optimizer and the biases gradients
     *
     * @param biases The biases that should be updated
     * @param biasesGrad The biases gradient
     *
     * The function will return void, since it only performs an update on the biases passed
     */
    virtual void updateBiases(Eigen::MatrixXd &biases, const Eigen::MatrixXd &biasesGrad) = 0;

    /**
     * @brief This function's purpose is to provide an interface to perform updates for the Optimizers from within the network
     *
     * @param size The size (for now it's only used to init the layer's in the Adam optimizer)
     *
     * This function returns nothing
     */
    virtual void insiderInit(size_t size) = 0;
  };
}