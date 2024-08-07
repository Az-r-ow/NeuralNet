#pragma once

#include <Eigen/Core>
#include <cmath>

#include "Optimizer.hpp"

namespace NeuralNet {
/**
 * Adam optimizer
 */
class Adam : public Optimizer {
 public:
  /**
   * Adam is an optimization algorithm that can be used instead of the classical
   * stochastic gradient descent procedure to update network weights
   * iteratively.
   *
   * @param alpha Learning rate
   * @param beta1 Exponential decay rate for the first moment estimates
   * @param beta2 Exponential decay rate for the second moment estimates
   * @param epsilon A small constant for numerical stability
   */
  Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999,
       double epsilon = 10E-8)
      : Optimizer(alpha) {
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
  };

  ~Adam() override = default;

  void updateWeights(Eigen::MatrixXd &weights,
                     const Eigen::MatrixXd &weightsGrad) override {
    this->update(weights, weightsGrad, mWeights[cl], vWeights[cl]);
  };

  void updateBiases(Eigen::MatrixXd &biases,
                    const Eigen::MatrixXd &biasesGrad) override {
    this->update(biases, biasesGrad, mBiases[cl], vBiases[cl]);
    this->setCurrentL();
  };

  template <typename Derived1, typename Derived2>
  void update(Eigen::MatrixBase<Derived1> &param,
              const Eigen::MatrixBase<Derived2> &gradients,
              Eigen::MatrixBase<Derived1> &m, Eigen::MatrixBase<Derived1> &v) {
    assert(param.rows() == gradients.rows() &&
           param.cols() == gradients.cols());

    // increment time step
    t = t + 1;

    if (m.rows() == 0 || m.cols() == 0) {
      // Initialize moment matrices m and v
      m = Eigen::MatrixBase<Derived1>::Zero(param.rows(), param.cols());
      v = Eigen::MatrixBase<Derived1>::Zero(param.rows(), param.cols());
    }

    assert(gradients.rows() == m.rows() && gradients.cols() == m.cols());
    assert(gradients.rows() == v.rows() && gradients.cols() == v.cols());

    // update biased first moment estimate
    m = (beta1 * m).array() + ((1 - beta2) * gradients.array()).array();

    // updated biased second raw moment estimate
    v = (beta2 * v).array() +
        ((1 - beta2) * (gradients.array() * gradients.array())).array();

    // compute bias-corrected first moment estimate
    double beta1_t = std::pow(beta1, t);

    // compute bias-corrected second raw moment estimate
    double beta2_t = std::pow(beta2, t);

    double alpha_t = alpha * (sqrt(1 - beta2_t) / (1 - beta1_t));

    // update param
    param =
        param.array() - alpha_t * (m.array() / (v.array().sqrt() + epsilon));
  }

 private:
  double beta1;
  double beta2;
  double epsilon;
  int t = 0;
  int cl;  // Current layer (should be initialized to the total number of layers
           // - 0)
  int ll;  // Last layer (should also be initialized to numLayers - 1)
  std::vector<Eigen::MatrixXd> mWeights;  // First-moment vector for weights
  std::vector<Eigen::MatrixXd> vWeights;  // Second-moment vector for weights
  std::vector<Eigen::MatrixXd> mBiases;   // First-moment vector for biases
  std::vector<Eigen::MatrixXd> vBiases;   // Second-moment vector for biases

  void insiderInit(size_t numLayers) override {
    cl = numLayers - 1;
    ll = numLayers - 1;

    Eigen::MatrixXd dotMatrix = Eigen::MatrixXd::Zero(0, 0);

    for (int i = mWeights.size(); i < numLayers; i++) {
      mWeights.push_back(dotMatrix);
      vWeights.push_back(dotMatrix);
      mBiases.push_back(dotMatrix);
      vBiases.push_back(dotMatrix);
    };
  }

  void setCurrentL() {
    // If current layer is the first layer set it to the last layer
    cl = cl == 1 ? ll : cl - 1;
  }
};
}  // namespace NeuralNet