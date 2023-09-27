#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include "Layer.hpp"
#include "utils/Formatters.hpp"
#include "utils/Gauge.hpp"
#include "interfaces/Optimizer.hpp"
#include "optimizers/optimizers.hpp"
#include "losses/losses.hpp"

namespace NeuralNet
{
    class Network
    {
    public:
        Network(double alpha = 0.001);

        /**
         * @brief Method that sets up the model's hyperparameters
         *
         * @param optimizer An Optimizer's child class
         * @param epochs The number of epochs
         * @param loss The loss function
         */
        void setup(const Optimizer &optimizer, int epochs = 10, LOSS loss = LOSS::QUADRATIC);

        /**
         * @brief Method to add a layer to the network
         *
         * @param layer the layer to add to the model it should be of type Layer
         */
        void addLayer(Layer &layer);

        /**
         * @brief This method will set the batch size of the network during training
         *
         * @param batchSize An integer > 0 that represents the batch size
         */
        void setBatchSize(int batchSize);

        /**
         * @brief This method will set the network's loss function
         *
         * @param loss The loss function (choose from the list of LOSS enums)
         */
        void setLoss(LOSS loss);

        /**
         * @brief This method will return the Layer residing at the specified index
         *
         * @param index The index from which to fetch the layer
         *
         * @return Layer at specified index
         */
        Layer getLayer(int index) const;

        /**
         * @brief This method will return the output layer (the last layer of the network)
         *
         * @return The output Layer
         */
        Layer getOutputLayer() const;

        /**
         * @brief This method will get you the number of layers currently in the Network
         *
         * @return An integer representing the number of layers in the Network
         */
        int getNumLayers() const;

        /**
         * @brief This method will Train the model with the given inputs and labels
         *
         * @param inputs The inputs that will be passed to the model
         * @param labels The labels that represent the expected outputs of the model
         *
         * @return The last training's loss
         */
        double train(std::vector<std::vector<double>> inputs, std::vector<double> labels);

        /**
         * @brief This model will try to make predictions based off the inputs passed
         *
         * @param inputs The inputs that will be passed through the network
         *
         * @return This method will return a vector of all the predictions made for each input
         */
        std::vector<double> predict(std::vector<std::vector<double>> inputs);
        ~Network();

    private:
        std::vector<Layer> layers;
        int cp = 0, tp = 0; // Correct Predictions, Total Predictions
        double alpha;       // Learning rate
        double loss = 1;    // Loss
        int batchSize = 50; // Default batch size
        int epochs;
        double (*cmpLoss)(const MatrixXd &, const Labels &);
        MatrixXd (*cmpGradient)(const MatrixXd &, const Labels &);
        SGD defaultOptimizer = SGD(alpha);
        Optimizer &optimizer = defaultOptimizer;

        MatrixXd forwardProp(std::vector<double> inputs);
        void backProp(MatrixXd grad);
        double computeAccuracy(int predicted, int label);
        MatrixXd nullifyGradient();
    };
}
