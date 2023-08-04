#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <Eigen/Core>
#include "Functions.hpp"
#include "utils/Enums.hpp"
#include "utils/Typedefs.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

class Network;

class Layer
{

    friend class Network;

public:
    Layer(int nNeurons, Activation activation = RELU, WeightInit weightInit = RANDOM, int bias = 0);
    void initWeights(int numCols);
    void setActivation(Activation activation);
    void feedInputs(vector<double> inputs);
    void feedInputs(MatrixXd inputs);
    int getNumNeurons() const;
    MatrixXd getOutputs();
    void printWeights();
    void printOutputs();
    ~Layer();

private:
    MatrixXd biases;
    WeightInit weightInit;
    MatrixXd outputs;
    MatrixXd weights;
    double (*activate)(double);

    void computeOutputs(MatrixXd inputs);
    void setOutputs(vector<double> outputs); // used for input layer

    /* Weight init */
    static void
    randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
    static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
};
