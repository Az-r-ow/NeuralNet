#include "Flatten.hpp"

using namespace NeuralNet;

Flatten::Flatten(std::tuple<int, int> inputShape, ACTIVATION activation, WEIGHT_INIT weightInit, int bias) : Layer(std::get<0>(inputShape) * std::get<1>(inputShape), activation, weightInit, bias)
{
  this->inputShape = inputShape;
}

void Flatten::feedInputs(std::vector<std::vector<std::vector<double>>> inputs)
{
  // Flatten the vectors
  std::vector<std::vector<double>> inputs1d(inputs.size());
  for (const std::vector<std::vector<double>> &input : inputs)
  {
    std::vector<double> flattenedInput = flatten2DVector(input, std::get<0>(inputShape), std::get<1>(inputShape));
    inputs1d.insert(inputs1d.end(), input.begin(), input.end());
  }

  const int numRows = inputs.size();
  const int numCols = inputs1d[0].size();

  // One last flatten to get all the inputs in one vector
  std::vector<double> flattenedInputs = flatten2DVector(inputs1d, numRows, numCols);

  this->setOutputs(Eigen::Map<Eigen::MatrixXd>(flattenedInputs.data(), numRows, numCols));
}

Flatten::~Flatten() {}