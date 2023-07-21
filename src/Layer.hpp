#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <Eigen/Core>
#include "Functions.hpp"
#include "utils/Enums.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

class Layer
{

public:
    Layer(int nNeurons, Activation activation = RELU, WeightInit weightInit = RANDOM, int bias = 0);
    void initWeights(int numRows);
    void setActivation(Activation activation);
    int getNumNeurons() const;
    void printWeights();
    void printOutputs();
    ~Layer();

private:
    int bias;
    WeightInit weightInit;
    VectorXd outputs;
    MatrixXd weights;
    double (*activate)(double);

    /* Weight init */
    static void randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
    static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
};
