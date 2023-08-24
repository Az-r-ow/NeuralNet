#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <Eigen/Dense>
#include "utils/Functions.hpp"
#include "utils/Enums.hpp"
#include "utils/Typedefs.hpp"
#include "activations/Sigmoid.hpp"
#include "activations/Relu.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

namespace NeuralNet
{
    class Network;

    class Layer
    {

        friend class Network;

    public:
        Layer(int nNeurons, ActivationName activationFunc = ActivationName::SIGMOID, WeightInit weightInit = WeightInit::RANDOM, int bias = 0);
        int getNumNeurons() const;
        MatrixXd getWeights() const;
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
        MatrixXd (*diff)(MatrixXd &);

        void initWeights(int numCols);
        void setActivation(ActivationName activation);
        void feedInputs(vector<double> inputs);
        void feedInputs(MatrixXd inputs);
        void computeOutputs(MatrixXd inputs);
        void setOutputs(vector<double> outputs); // used for input layer

        static void
        randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
        static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
    };
}
