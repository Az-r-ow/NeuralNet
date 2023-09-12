#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <Eigen/Dense>
#include "utils/Functions.hpp"
#include "utils/Enums.hpp"
#include "utils/Typedefs.hpp"
#include "activations/activations.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNet
{
    class Network;

    class Layer
    {

        friend class Network;

    public:
        Layer(int nNeurons, ACTIVATION activationFunc = ACTIVATION::SIGMOID, WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0);
        int getNumNeurons() const;
        MatrixXd getWeights() const;
        MatrixXd getOutputs();
        void printWeights();
        void printOutputs();
        ~Layer();

    private:
        MatrixXd biases;
        WEIGHT_INIT weightInit;
        MatrixXd outputs;
        MatrixXd weights;
        MatrixXd (*activate)(const MatrixXd &);
        MatrixXd (*diff)(const MatrixXd &);

        void initWeights(int numCols);
        void setActivation(ACTIVATION activation);
        void feedInputs(std::vector<double> inputs);
        void feedInputs(MatrixXd inputs);
        void computeOutputs(MatrixXd inputs);
        void setOutputs(std::vector<double> outputs); // used for input layer

        static void
        randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
        static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
    };
}
