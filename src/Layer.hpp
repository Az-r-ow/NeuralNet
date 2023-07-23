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

class Layer
{

public:
    Layer(int nNeurons, Activation activation = RELU, WeightInit weightInit = RANDOM, int bias = 0);
    void initWeights(int numCols);
    void setActivation(Activation activation);
    void feedInputs(vector<double> inputs);
    void feedInputs(Matrix1d inputs);
    int getNumNeurons() const;
    Matrix1d getOutputs() const;
    void printWeights();
    void printOutputs();
    ~Layer();

private:
    int bias;
    WeightInit weightInit;
    Matrix1d outputs;
    MatrixXd weights;
    double (*activate)(double);

    void computeOutputs(Matrix1d inputs);
    void setOutputs(Matrix1d outputs);

    /* Weight init */
    static void
    randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
    static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
};
