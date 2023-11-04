#include "Flatten.hpp"

using namespace NeuralNet;

Flatten::Flatten(std::tuple<int, int> inputShape, ACTIVATION activation, WEIGHT_INIT weightInit, int bias) : Layer(std::get<0>(inputShape) * std::get<1>(inputShape), activation, weightInit, bias)
{
  this->inputShape = inputShape;
}

void Flatten::feedInputs(std::vector<std::vector<std::vector<double>>> inputs)
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

Flatten::~Flatten() {}