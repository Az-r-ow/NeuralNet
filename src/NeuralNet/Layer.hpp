#pragma once

#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include <Eigen/Dense>
#include <cereal/access.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/base_class.hpp>
#include "utils/Functions.hpp"
#include "utils/Enums.hpp"
#include "utils/Typedefs.hpp"
#include "activations/activations.hpp"
#include "utils/Serialize.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNet
{
    class Network;

    class Layer
    {

        friend class Network;

    public:
        Layer(int nNeurons, ACTIVATION activation = ACTIVATION::SIGMOID, WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0);
        Layer() {}
        /**
         * @brief This method get the number of neurons actually in the layer
         *
         * @return The number of neurons in the layer
         */
        int getNumNeurons() const;

        /**
         * @brief This method method gets the layer's weights
         *
         * @return an Eigen::MatrixXd representing the weights
         */
        MatrixXd getWeights() const;

        /**
         * @brief This method get the layer's outputs
         *
         * @return an Eigen::MatrixXd representing the layer's outputs
         */
        MatrixXd getOutputs() const;

        /**
         * @brief Method to print layer's weights
         */
        void printWeights();

        /**
         * @brief Method to print layer's outputs
         */
        void printOutputs();
        ~Layer();

    private:
        // non-public serialization
        friend class cereal::access;

        MatrixXd biases;
        WEIGHT_INIT weightInit;
        MatrixXd outputs;
        MatrixXd weights;
        ACTIVATION activation;
        MatrixXd (*activate)(const MatrixXd &);
        MatrixXd (*diff)(const MatrixXd &);

        void initWeights(int numCols);
        void feedInputs(std::vector<double> inputs);
        void feedInputs(MatrixXd inputs);
        void computeOutputs(MatrixXd inputs);
        void setActivation(ACTIVATION activation);
        void setOutputs(std::vector<double> outputs); // used for input layer

        // Necessary function for serializing Layer
        template <class Archive>
        void save(Archive &archive) const
        {
            archive(weights, outputs, biases, activation);
        };

        template <class Archive>
        void load(Archive &archive)
        {
            archive(weights, outputs, biases, activation);
            setActivation(activation);
        }

        static void
        randomWeightInit(MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
        static void randomDistWeightInit(MatrixXd *weightsMatrix, double mean, double stddev);
    };
}
