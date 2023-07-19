#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <Eigen/Core>
#include "Functions.hpp"
#include "utils/Enums.hpp"

using Eigen::MatrixXd;
using std::string;

class Layer
{
public:
    Layer(int nNeurons, Activation activation = RELU, WeightInit weightInit = RANDOM, int bias = 0);
    void initWeights(int numRows);
    int getNumNeurons() const;
    void printWeights();
    ~Layer();

private:
    int numNeurons;
    int bias;
    WeightInit weightInit;
    MatrixXd weights;

    /* Weight init */
    static void randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
    static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
};
