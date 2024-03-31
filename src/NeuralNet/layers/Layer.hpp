#pragma once

#include <Eigen/Dense>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>

#include "Network.hpp"
#include "activations/activations.hpp"
#include "utils/Enums.hpp"
#include "utils/Functions.hpp"
#include "utils/Serialize.hpp"

namespace NeuralNet {
// Should be updated when a new layer type is added
enum class LayerType { DEFAULT, DENSE, FLATTEN, DROPOUT };

class Layer {
  friend class Network;

 public:
  Layer(){};

  /**
   * @brief This method get the layer's outputs
   *
   * @return an Eigen::Eigen::MatrixXd  representing the layer's outputs
   */
  Eigen::MatrixXd getOutputs() const { return outputs; };

  /**
   * @brief This method get the number of neurons actually in the layer
   *
   * @return The number of neurons in the layer
   */
  int getNumNeurons() const { return nNeurons; };

  /**
   * @brief Method to print layer's outputs
   */
  void printOutputs() {
    std::cout << this->outputs << "\n";
    return;
  };

  /**
   * @brief This method is used to feed the inputs to the layer
   *
   * @param inputs A vector of doubles representing the inputs (features)
   *
   * @return an Eigen::MatrixXd representing the outputs of the layer
   */
  virtual Eigen::MatrixXd feedInputs(std::vector<double> inputs) {
    return this->feedInputs(Eigen::MatrixXd::Map(&inputs[0], inputs.size(), 1));
  };

  /**
   * @brief This method is used to feed the inputs to the layer
   *
   * @param inputs An Eigen::MatrixXd representing the inputs (features)
   *
   * @return an Eigen::MatrixXd representing the outputs of the layer
   */
  virtual Eigen::MatrixXd feedInputs(Eigen::MatrixXd inputs) = 0;

  /**
   * @brief This method is used to feed the inputs to the layer
   *
   * @param inputs A vector of vectors of doubles representing the inputs
   * (features)
   *
   * @return void
   */
  virtual void feedInputs(
      std::vector<std::vector<std::vector<double>>> inputs) {
    assert(false &&
           "Cannot feed 3d vectors, a Flatten layer could do it though");
    return;
  };

  /**
   * Returns the layer type as string
   *
   * @note Returns "Base" for base class `Layer` and "Unknown" if no type
   * specified.
   */
  const std::string typeStr() {
    static const std::map<LayerType, std::string> typeMap = {
        {LayerType::DEFAULT, "Base"},
        {LayerType::DENSE, "Dense"},
        {LayerType::FLATTEN, "Flatten"},
        {LayerType::DROPOUT, "Dropout"}};

    auto it = typeMap.find(this->type);
    if (it != typeMap.end()) return it->second;

    return "Unknown";
  };

  virtual ~Layer(){};

 private:
  // non-public serialization
  friend class cereal::access;

  // Necessary function for serializing Layer
  template <class Archive>
  void save(Archive &archive) const {
    archive(outputs, type);
  };

  template <class Archive>
  void load(Archive &archive) {
    archive(outputs, type);
  }

 protected:
  int nNeurons;
  Eigen::MatrixXd outputs;
  LayerType type = LayerType::DEFAULT;
  bool trainingOnly = false;  // If true skip during inferences

  /**
   * @param outputs the outputs to store
   */
  void setOutputs(Eigen::MatrixXd outputs)  // used for the Flatten Layer
  {
    this->outputs = outputs;
  };

  /**
   * @brief This method is used to set the outputs of the layer
   *
   * @param outputs A vector of doubles representing the outputs of the layer
   *
   * @return void
   *
   * @note This method is used for the input layer (the first layer of the
   * network)
   */
  void setOutputs(std::vector<double> outputs) {
    assert(outputs.size() == nNeurons);
    this->outputs =
        Eigen::MatrixXd ::Map(&outputs[0], this->getNumNeurons(), 1);
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
  virtual Eigen::MatrixXd computeOutputs(Eigen::MatrixXd inputs) = 0;

  /**
   * This function will be used to properly initialize the Layer
   * It's being done like this because of the number of neurons of the previous
   * layer that's unkown prior
   */
  virtual void init(int args){};

  Layer(std::tuple<int, int> inputShape)
      : nNeurons(std::get<0>(inputShape) *
                 std::get<1>(inputShape)){};  // Used in Flatten layer
};
}  // namespace NeuralNet

CEREAL_REGISTER_TYPE(NeuralNet::Layer);