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
         * @return an Eigen::Eigen::MatrixXd  representing the weights
         */
        Eigen::MatrixXd getWeights() const;

        /**
         * @brief This method get the layer's outputs
         *
         * @return an Eigen::Eigen::MatrixXd  representing the layer's outputs
         */
        Eigen::MatrixXd getOutputs() const;

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

        double bias;
        int nNeurons; // Number of neurons
        Eigen::MatrixXd biases;
        WEIGHT_INIT weightInit;
        Eigen::MatrixXd outputs;
        Eigen::MatrixXd weights;
        ACTIVATION activation;
        Eigen::MatrixXd (*activate)(const Eigen::MatrixXd &);
        Eigen::MatrixXd (*diff)(const Eigen::MatrixXd &);

        void init(int numCols);
        void feedInputs(std::vector<double> inputs);
        void feedInputs(Eigen::MatrixXd inputs);
        virtual void feedInputs(std::vector<std::vector<std::vector<double>>> inputs);
        void computeOutputs(Eigen::MatrixXd inputs);
        void setActivation(ACTIVATION activation);

        // Necessary function for serializing Layer
        template <class Archive>
        void save(Archive &archive) const
        {
            archive(nNeurons, weights, outputs, biases, activation);
        };

        template <class Archive>
        void load(Archive &archive)
        {
            archive(nNeurons, weights, outputs, biases, activation);
            setActivation(activation);
        }

        static void
        randomWeightInit(Eigen::MatrixXd *weightsMatrix, double min = -1.0, double max = 1.0);
        static void randomDistWeightInit(Eigen::MatrixXd *weightsMatrix, double mean, double stddev);

    protected:
        void setOutputs(std::vector<double> outputs); // used for input layer
        void setOutputs(Eigen::MatrixXd outputs);     // used for the Flatten Layer
    };
}
