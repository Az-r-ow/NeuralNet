#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <cereal/access.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <Eigen/Dense>
#include "utils/Functions.hpp"
#include "utils/Serialize.hpp"
#include "utils/Enums.hpp"
#include "activations/activations.hpp"
#include "Network.hpp"

namespace NeuralNet
{
  // Should be updated when a new layer type is added
  enum class LayerType
  {
    DEFAULT,
    DENSE,
    FLATTEN
  };

  class Layer
  {
    friend class Network;

  public:
    Layer(int nNeurons, ACTIVATION activation = ACTIVATION::SIGMOID, WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0)
    {
      this->bias = bias;
      this->nNeurons = nNeurons;
      this->weightInit = weightInit;
      this->activation = activation;
      this->setActivation(activation);
    };

    /**
     * @brief This method get the number of neurons actually in the layer
     *
     * @return The number of neurons in the layer
     */
    int getNumNeurons() const
    {
      return nNeurons;
    };

    /**
     * @brief This method method gets the layer's weights
     *
     * @return an Eigen::Eigen::MatrixXd  representing the weights
     */
    Eigen::MatrixXd getWeights() const
    {
      return weights;
    }

    /**
     * @brief This method get the layer's outputs
     *
     * @return an Eigen::Eigen::MatrixXd  representing the layer's outputs
     */
    Eigen::MatrixXd getOutputs() const
    {
      return outputs;
    };

    /**
     * @brief Method to print layer's weights
     */
    void printWeights()
    {
      std::cout << this->weights << "\n";
      return;
    };

    /**
     * @brief Method to print layer's outputs
     */
    void printOutputs()
    {
      std::cout << this->outputs << "\n";
      return;
    };

    ~Layer(){};

  private:
    // non-public serialization
    friend class cereal::access;

    double bias;
    Eigen::MatrixXd biases;
    WEIGHT_INIT weightInit;
    Eigen::MatrixXd outputs;
    Eigen::MatrixXd weights;
    ACTIVATION activation;
    Eigen::MatrixXd (*activate)(const Eigen::MatrixXd &);
    Eigen::MatrixXd (*diff)(const Eigen::MatrixXd &);

    /**
     * This function will be used to properly initialize the Layer
     * It's being done like this because of the number of neurons of the previous layer that's unkown prior
     */
    void init(int numRows)
    {
      // First and foremost init the biases and the outputs
      double mean = 0, stddev = 0;
      this->weights = Eigen::MatrixXd::Zero(numRows, nNeurons);

      // This is going to be used for testing
      if (this->weightInit == WEIGHT_INIT::CONSTANT)
      {
        this->weights = Eigen::MatrixXd::Constant(numRows, nNeurons, 1);
        return;
      }

      // calculate mean and stddev based on init algo
      switch (this->weightInit)
      {
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
      this->weightInit == WEIGHT_INIT::RANDOM ? randomWeightInit(&(this->weights), -1, 1) : randomDistMatrixInit(&(this->weights), mean, stddev);
    }

    virtual void feedInputs(std::vector<double> inputs)
    {
      this->feedInputs(Eigen::MatrixXd::Map(&inputs[0], inputs.size(), 1));
      return;
    };

    // todo: return the outputs directly
    virtual void feedInputs(Eigen::MatrixXd inputs)
    {
      // Layer is "input" layer
      if (weights.rows() == 0)
      {
        this->setOutputs(inputs);
        return;
      }

      inputs = inputs.cols() == weights.rows() ? inputs : inputs.transpose();

      assert(inputs.cols() == weights.rows());
      this->computeOutputs(inputs);
      return;
    };

    virtual void feedInputs(std::vector<std::vector<std::vector<double>>> inputs)
    {
      assert(false && "Cannot feed 3d vectors, a Flatten layer could do it though");
      return;
    };

    void computeOutputs(Eigen::MatrixXd inputs)
    {
      // Initialize the biases based on the input's size
      if (biases.rows() == 0 && biases.cols() == 0)
      {
        biases = Eigen::MatrixXd::Constant(1, nNeurons, bias);
      }

      // Weighted sum
      Eigen::MatrixXd wSum = inputs * weights;

      wSum.rowwise() += biases.row(0);

      outputs = activate(wSum);
      return;
    };

    void setActivation(ACTIVATION activation)
    {
      if (type == LayerType::FLATTEN)
      {
        this->activate = Activation::activate;
        this->diff = Activation::diff;
        return;
      }

      switch (activation)
      {
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

    // Necessary function for serializing Layer
    template <class Archive>
    void save(Archive &archive) const
    {
      archive(nNeurons, weights, outputs, biases, activation, type);
    };

    template <class Archive>
    void load(Archive &archive)
    {
      archive(nNeurons, weights, outputs, biases, activation, type);
      setActivation(activation);
    }

  protected:
    Layer(){};                                                                                              // Necessary for serialization
    Layer(std::tuple<int, int> inputShape) : nNeurons(std::get<0>(inputShape) * std::get<1>(inputShape)){}; // Used in Flatten layer

    int nNeurons; // Number of neurons
    LayerType type = LayerType::DEFAULT;

    void setOutputs(std::vector<double> outputs) // used for input layer
    {
      assert(outputs.size() == nNeurons);
      this->outputs = Eigen::MatrixXd ::Map(&outputs[0], this->getNumNeurons(), 1);
    };
    void setOutputs(Eigen::MatrixXd outputs) // used for the Flatten Layer
    {
      this->outputs = outputs;
    };
  };
}

CEREAL_REGISTER_TYPE(NeuralNet::Layer);