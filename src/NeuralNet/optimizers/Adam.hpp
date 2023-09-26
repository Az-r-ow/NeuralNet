#pragma once

#include "optimizers/optimizers.hpp"

namespace NeuralNet
{
  /**
   * Adam optimizer
   */
  class Adam : public Optimizer
  {
  public:
    Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 10E-8) : Optimizer(alpha)
    {
      this->beta1 = beta1;
      this->beta2 = beta2;
      this->epsilon = epsilon;
    };

    void updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &weightsGrad) const override
    {
      weights = weights.array() - (this->alpha * weightsGrad.transpose()).array();
    };

    void updateBiases(Eigen::MatrixXd &biases, const Eigen::MatrixXd &biasesGrad) const override
    {
      biases = biases.array() - (this->alpha * biasesGrad.transpose()).array();
    };

    ~Adam() override = default;

  private:
    double beta1;
    double beta2;
    double epsilon;
    int t;
    Eigen::MatrixXd mWeights; // First-moment vector for weights
    Eigen::MatrixXd vWeights; // Second-moment vector for weights
    Eigen::VectorXf mBiases;  // First-moment vector for biases
    Eigen::VectorXf vBiases;  // Second-moment vector for biases

    template <typename Derived1, typename Derived2>
    void update(Eigen::MatrixBase<Derived1> &param, const Eigen::MatrixBase<Derived2> &gradients, Eigen::MatrixBase<Derived1> &m, Eigen::MatrixBase<Derived1> &v)
    {
      assert(param.rows() != gradients.rows() || param.cols() != gradients.cols());

      if (m.rows() == 0 || m.cols() == 0)
      {
        // Initialize moment matrices m and v
        m = Eigen::MatrixBase<Derived1>::Zero(param.rows(), param.cols());
        v = Eigen::MatrixBase<Derived1>::Zero(param.rows(), param.cols());
      }

      // update biased first moment estimate
      m = (beta1 * m).array() + ((1 - beta2) * gradients).array();

      // updated biased second raw moment estimate
      v = (beta2 * v).array() + ((1 - beta2) * (gradients.array() * gradients.array())).array();

      // compute bias-corrected first moment estimate
      double beta1t = std::pow(beta1, t);
      Eigen::MatrixBase<Derived1> mHat = m.array() / (1 - beta1t);

      // compute bias-corrected second raw moment estimate
      double beta2t = std::pow(beta2, t);
      Eigen::MatrixBase<Derived1> vHat = v.array() / (1 - beta2t);

      // update param
      param = param.array() - alpha * mHat / (vHat.array().sqrt() + epsilon).array();

      // increment time step
      t++;
    }
  };
}