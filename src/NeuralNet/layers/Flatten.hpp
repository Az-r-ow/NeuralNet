#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <tuple>

#include "Layer.hpp"
#include "utils/Functions.hpp"
#include "utils/Serialize.hpp"

namespace NeuralNet {
class Flatten : public Layer {
 public:
  /**
   * @brief Construct a new Flatten object.
   *
   * This type of Layer is perfect as an input layer since its sole purpose is
   * to flatten a given input
   *
   * @param inputShape The shape of the input to be flattened
   */
  Flatten(std::tuple<int, int> inputShape)
      : Layer(inputShape), inputShape(inputShape) {
    type = LayerType::FLATTEN;
  };

  /**
   * @brief This method flattens a 3D vector into a 2D Eigen::MatrixXd
   *
   * @param inputs The 3D vector to be flattened
   * @return Eigen::MatrixXd The flattened 2D Eigen::MatrixXd
   */
  Eigen::MatrixXd flatten(
      std::vector<std::vector<std::vector<double>>> inputs) {
    int rows = std::get<0>(inputShape);
    int cols = std::get<1>(inputShape);

    // Flatten the vectors
    std::vector<double> flatInputs;
    for (const std::vector<std::vector<double>> &input : inputs) {
      std::vector<double> flattenedInput = flatten2DVector(input, rows, cols);
      flatInputs.insert(flatInputs.end(), flattenedInput.begin(),
                        flattenedInput.end());
    }

    const int numRows = inputs.size();
    const int numCols = rows * cols;
    return Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        flatInputs.data(), numRows, numCols);
  };

  ~Flatten(){};

 private:
  // non-public serialization
  friend class cereal::access;

  Flatten(){};  // Necessary for serialization

  std::tuple<int, int> inputShape;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::base_class<Layer>(this), inputShape);
  }

  void feedInputs(
      std::vector<std::vector<std::vector<double>>> inputs) override {
    Eigen::MatrixXd flattenedInputs = this->flatten(inputs);
    this->setOutputs(flattenedInputs);
  };

  Eigen::MatrixXd feedInputs(Eigen::MatrixXd inputs) override {
    this->setOutputs(inputs);
    return outputs;
  };

  Eigen::MatrixXd computeOutputs(Eigen::MatrixXd inputs) override {
    return outputs;
  }
};
}  // namespace NeuralNet

namespace cereal {
template <class Archive>
struct specialize<Archive, NeuralNet::Flatten,
                  cereal::specialization::member_serialize> {};
}  // namespace cereal

CEREAL_REGISTER_TYPE(NeuralNet::Flatten);

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNet::Layer, NeuralNet::Flatten);