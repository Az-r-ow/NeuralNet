#pragma once

#include <algorithm>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <iterator>
#include <random>
#include <tuple>

#include "Layer.hpp"

namespace NeuralNet {
class Dropout : public Layer {
 public:
  float rate, scaleRate;
  unsigned int seed;
  Eigen::MatrixXd mask;

  /**
   * @brief The Dropout layer randomly sets input units to 0 with a frequency of
   * rate at each step during training time, which helps prevent overfitting.
   * Inputs not set to 0 are scaled up by 1 / (1 - rate) such that the sum over
   * all inputs is unchanged.
   *
   * @param rate Frequency of units set to 0
   * @param seed An integer to use as a random seed
   */
  Dropout(float rate, unsigned int seed = 0) : rate(rate), seed(seed) {
    assert(rate < 1 && rate > 0);
    this->type = LayerType::DROPOUT;
    this->trainingOnly = true;  // Training only layer
    this->scaleRate = 1.0 / (1.0 - rate);
  };

  /**
   * @brief Dropout layer slug
   */
  std::string getSlug() const override {
    return slug + removeTrailingZeros(std::to_string(rate));
  }

  /**
   * @brief This method is used to feed the inputs to the layer
   *
   * @param inputs An Eigen::MatrixXd representing the inputs (features)
   *
   * @return an Eigen::MatrixXd representing the outputs of the layer
   */
  Eigen::MatrixXd feedInputs(Eigen::MatrixXd inputs) override {
    return this->computeOutputs(inputs);
  };

 private:
  std::vector<std::tuple<int, int>> coordinates;
  std::string slug = "do";

  // non-public serialization
  friend class cereal::access;

  Dropout(){};  // Necessary for serialization

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<Layer>(this), seed, rate);
  }

  /**
   * @brief Check seed before generating one randomly
   *
   * @return seed
   */
  unsigned int getSeed() {
    if (seed != 0) return seed;
    std::random_device rd;
    return rd();
  };

 protected:
  /**
   * @param numNeurons Number of neurons of the previous layers
   */
  void init(int numNeurons) override { this->nNeurons = numNeurons; };

  /**
   * @brief Drop some of the inputs randomly at the given rate
   *
   * @param inputs A matrix representing the inputs (features)
   *
   * @return Inputs with some dropped values (zero-ed values) randomly
   */
  Eigen::MatrixXd computeOutputs(Eigen::MatrixXd inputs) override {
    int rows = inputs.rows();
    int cols = inputs.cols();
    int numCoord = rows * cols;  // Number of coordinates
    mask = Eigen::MatrixXd::Constant(rows, cols, 1);

    seed = getSeed();
    std::mt19937 gen(seed);
    const int num_zeros = static_cast<int>(numCoord * (1.0 - rate));

    std::vector<std::tuple<int, int>> randCoordinates;

    randCoordinates.reserve(num_zeros);

    if (!coordinates.size() || coordinates.size() != (numCoord)) {
      coordinates.clear();
      coordinates.reserve(numCoord);

      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          coordinates.emplace_back(std::make_tuple(i, j));
        }
      }
    }

    // Randomly select tuples from coordinates
    std::sample(coordinates.begin(), coordinates.end(),
                std::back_inserter(randCoordinates), num_zeros, gen);

    for (std::tuple<int, int>& coord : randCoordinates) {
      mask(std::get<0>(coord), std::get<1>(coord)) = 0;
    }

    outputs = (inputs.array() * mask.array()) * scaleRate;

    return outputs;
  };
};
}  // namespace NeuralNet

namespace cereal {
template <class Archive>
struct specialize<Archive, NeuralNet::Dropout,
                  cereal::specialization::member_serialize> {};
}  // namespace cereal

CEREAL_REGISTER_TYPE(NeuralNet::Dropout);

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNet::Layer, NeuralNet::Dropout);