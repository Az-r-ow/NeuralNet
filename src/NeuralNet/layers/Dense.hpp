#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

#include "Layer.hpp"

namespace NeuralNet {
class Dense : public Layer {
 public:
  Dense(int nNeurons, ACTIVATION activation = ACTIVATION::SIGMOID,
        WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0) {
    type = LayerType::DENSE;
    this->bias = bias;
    this->nNeurons = nNeurons;
    this->weightInit = weightInit;
    this->activation = activation;
    this->setActivation(activation);
  };

  /**
   * @brief This method gets the layer's weights
   *
   * @return an Eigen::MatrixXd  representing the weights
   */
  Eigen::MatrixXd getWeights() const { return weights; };

  /**
   * @brief Return the biases of the layer
   *
   * @return an Eigen::Matrix representing the biases
   */
  Eigen::MatrixXd getBiases() const { return biases; };

  /**
   * @brief This method get the layer's outputs
   *
   * @return an Eigen::Eigen::MatrixXd  representing the layer's outputs
   */
  Eigen::MatrixXd getOutputs() const { return outputs; };

  /**
   * @brief Method to print layer's weights
   */
  void printWeights() {
    std::cout << this->weights << "\n";
    return;
  };

  ~Dense(){};

 private:
  // non-public serialization
  friend class cereal::access;
  friend class Network;

  double bias;
  Eigen::MatrixXd biases;
  WEIGHT_INIT weightInit;
  Eigen::MatrixXd weights;
  Eigen::MatrixXd cachedWeights;
  Eigen::MatrixXd cachedBiases;
  ACTIVATION activation;
  Eigen::MatrixXd (*activate)(const Eigen::MatrixXd &);
  Eigen::MatrixXd (*diff)(const Eigen::MatrixXd &);

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::base_class<Layer>(this), nNeurons, biases, weights, activation);
  }

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::base_class<Layer>(this), nNeurons, biases, weights, activation);
    setActivation(activation);
  }

  /**
   * This function will be used to properly initialize the Layer
   * It's being done like this because of the number of neurons of the previous
   * layer that's unkown prior
   */
  void init(int numRows) override {
    // First and foremost init the biases and the outputs
    double mean = 0, stddev = 0;
    this->weights = Eigen::MatrixXd::Zero(numRows, nNeurons);

    // This is going to be used for testing
    if (this->weightInit == WEIGHT_INIT::CONSTANT) {
      this->weights = Eigen::MatrixXd::Constant(numRows, nNeurons, 1);
      return;
    }

    // calculate mean and stddev based on init algo
    switch (this->weightInit) {
      case WEIGHT_INIT::GLOROT:
        // sqrt(fan_avg)
        stddev = sqrt(static_cast<double>(2) / (numRows + nNeurons));
        break;
      case WEIGHT_INIT::HE:
        // sqrt(2/fan_in)
        stddev = sqrt(2.0 / numRows);
        break;
      case WEIGHT_INIT::LECUN:
        // sqrt(1/fan_in)
        stddev = sqrt(1.0 / numRows);
        break;
      default:
        break;
    }

    // Init the weights
    this->weightInit == WEIGHT_INIT::RANDOM
        ? randomWeightInit(&(this->weights), -1, 1)
        : randomDistMatrixInit(&(this->weights), mean, stddev);
  }

  /**
   * @brief This method is used to feed the inputs to the layer
   *
   * @param inputs An Eigen::MatrixXd representing the inputs (features)
   *
   * @return an Eigen::MatrixXd representing the outputs of the layer
   */
  virtual Eigen::MatrixXd feedInputs(Eigen::MatrixXd inputs) override {
    // Dense layer positioned as input layer
    if (weights.rows() == 0 && weights.cols() == 0) {
      setOutputs(inputs);
      return inputs;
    }

    inputs = inputs.cols() == weights.rows() ? inputs : inputs.transpose();

    assert(inputs.cols() == weights.rows());
    return this->computeOutputs(inputs);
  };

  /**
   * @brief This method is used to feed the inputs to the layer
   *
   * @param inputs A vector of vectors of doubles representing the inputs
   * (features)
   *
   * @return an Eigen::MatrixXd representing the computed outputs based on the
   * layer's parameters
   */
  Eigen::MatrixXd computeOutputs(Eigen::MatrixXd inputs) override {
    // Initialize the biases based on the input's size
    if (biases.rows() == 0 && biases.cols() == 0) {
      biases = Eigen::MatrixXd::Constant(1, nNeurons, bias);
    }

    // Weighted sum
    Eigen::MatrixXd wSum = inputs * weights;

    wSum.rowwise() += biases.row(0);

    outputs = activate(wSum);
    return outputs;
  };

  /**
   * @brief This method is used to set the activation function of the layer
   *
   * @param activation The activation function to be used
   *
   * @return void
   */
  void setActivation(ACTIVATION activation) {
    if (type == LayerType::FLATTEN) {
      this->activate = Activation::activate;
      this->diff = Activation::diff;
      return;
    }

    switch (activation) {
      case ACTIVATION::SIGMOID:
        this->activate = Sigmoid::activate;
        this->diff = Sigmoid::diff;
        break;
      case ACTIVATION::RELU:
        this->activate = Relu::activate;
        this->diff = Relu::diff;
        break;
      case ACTIVATION::SOFTMAX:
        this->activate = Softmax::activate;
        this->diff = Softmax::diff;
        break;
      /**
       * Add cases as I add activations
       */
      default:
        assert(false && "Activation not defined");
    }

    return;
  };

  Dense(){};  // Required for serialization
};
}  // namespace NeuralNet

CEREAL_REGISTER_TYPE(NeuralNet::Dense);

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNet::Layer, NeuralNet::Dense);
