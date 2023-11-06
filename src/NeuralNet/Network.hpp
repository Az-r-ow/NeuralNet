#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include <cereal/cereal.hpp> // for defer
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "Model.hpp"
#include "utils/Formatters.hpp"
#include "utils/Functions.hpp"
#include "utils/Gauge.hpp"
#include "optimizers/Optimizer.hpp"
#include "layers/Layer.hpp"
#include "layers/Flatten.hpp"
#include "layers/Dense.hpp"
#include "optimizers/optimizers.hpp"
#include "losses/losses.hpp"

namespace NeuralNet
{
    class Layer;

    class Network : public Model
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
        void setup(const std::shared_ptr<Optimizer> &optimizer, int epochs = 10, LOSS loss = LOSS::QUADRATIC);

        /**
         * @brief Method to add a layer to the network
         *
         * @param layer the layer to add to the model it should be of type Layer
         */
        void addLayer(std::shared_ptr<Layer> &layer);

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
        std::shared_ptr<Layer> getLayer(int index) const;

        /**
         * @brief This method will return the output layer (the last layer of the network)
         *
         * @return The output Layer
         */
        std::shared_ptr<Layer> getOutputLayer() const;

        /**
         * @brief This method will get you the number of layers currently in the Network
         *
         * @return A size_t representing the number of layers in the Network
         */
        size_t getNumLayers() const;

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
         * @brief This method will Train the model with the given inputs and labels
         *
         * @param inputs The inputs that will be passed to the model
         * @param labels The labels that represent the expected outputs of the model
         *
         * @return The last training's loss
         */
        double train(std::vector<std::vector<std::vector<double>>> inputs, std::vector<double> labels);

        /**
         * @brief This model will try to make predictions based off the inputs passed
         *
         * @param inputs The inputs that will be passed through the network
         *
         * @return This method will return the outputs of the neural network
         */
        Eigen::MatrixXd predict(std::vector<std::vector<double>> inputs);

        /**
         * @brief This model will try to make predictions based off the inputs passed
         *
         * @param inputs The inputs that will be passed through the network
         *
         * @return This method will return the outputs of the neural network
         */
        Eigen::MatrixXd predict(std::vector<std::vector<std::vector<double>>> inputs);

        ~Network();

    private:
        // non-public serialization
        friend class cereal::access;

        template <class Archive>
        void save(Archive &archive) const
        {
            archive(layers, lossFunc);
            archive.serializeDeferments();
        };

        template <class Archive>
        void load(Archive &archive)
        {
            archive(layers, lossFunc);
            setLoss(lossFunc);
        }

        std::vector<std::shared_ptr<Layer>> layers;
        LOSS lossFunc;      // Storing the loss function for serialization
        int cp = 0, tp = 0; // Correct Predictions, Total Predictions
        double alpha;       // Learning rate
        double loss = 1;    // Loss
        int batchSize = 50; // Default batch size
        int epochs;
        double (*cmpLoss)(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
        Eigen::MatrixXd (*cmpGradient)(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
        std::shared_ptr<Optimizer> optimizer;

        // The template are called D for dimensions eg : 2d 3d
        template <typename D1, typename D2>
        double trainingProcess(std::vector<D1> inputs, std::vector<D2> labels);
        Eigen::MatrixXd forwardProp(std::vector<std::vector<std::vector<double>>> inputs);
        Eigen::MatrixXd forwardProp(std::vector<std::vector<double>> inputs);
        Eigen::MatrixXd forwardProp(Eigen::MatrixXd inputs);
        void backProp(Eigen::MatrixXd grad);
        double computeAccuracy(int predicted, int label);
        void updateOptimizerSetup(size_t numLayers);
    };
}
