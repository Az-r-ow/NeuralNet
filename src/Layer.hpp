#pragma once

#include <cstdlib>
#include <string>
#include <Eigen/Core>
#include "Functions.hpp"

using Eigen::MatrixXd;
using std::string;

enum Activation
{
    RELU,
    SIGMOID
};

class Layer
{
public:
    Layer(int nNeurons, Activation activation = RELU, int bias = 0);
    void initWeights(int numRows);
    int getNumNeurons() const;
    void printWeights();
    ~Layer();

private:
    int numNeurons;
    int bias;
    MatrixXd weights;
};
