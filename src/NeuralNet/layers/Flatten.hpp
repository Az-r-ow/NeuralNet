#pragma once

#include <tuple>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include "Layer.hpp"
#include "utils/Functions.hpp"
#include "utils/Serialize.hpp"

namespace NeuralNet
{
  class Flatten : public Layer
  {
  public:
    Flatten(std::tuple<int, int> inputShape) : Layer(inputShape), inputShape(inputShape){};

    ~Flatten(){};

  private:
    // non-public serialization
    friend class cereal::access;

    Flatten(){}; // Necessary for serialization

    std::tuple<int, int> inputShape;

    template <class Archive>
    void serialize(Archive &ar)
    {
      ar(cereal::base_class<Layer>(this), inputShape);
    }

    void feedInputs(std::vector<std::vector<std::vector<double>>> inputs)
    {
      int rows = std::get<0>(inputShape);
      int cols = std::get<1>(inputShape);

      // Flatten the vectors
      std::vector<double> flatInputs(inputs.size());
      for (const std::vector<std::vector<double>> &input : inputs)
      {
        std::vector<double> flattenedInput = flatten2DVector(input, rows, cols);
        flatInputs.insert(flatInputs.end(), flattenedInput.begin(), flattenedInput.end());
      }

      const int numRows = inputs.size();
      const int numCols = rows * cols;

      this->setOutputs(Eigen::Map<Eigen::MatrixXd>(flatInputs.data(), numRows, numCols));
    }
  };
}

namespace cereal
{
  template <class Archive>
  struct specialize<Archive, NeuralNet::Flatten, cereal::specialization::member_serialize>
  {
  };
}

CEREAL_REGISTER_TYPE(NeuralNet::Flatten);

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNet::Layer, NeuralNet::Flatten);