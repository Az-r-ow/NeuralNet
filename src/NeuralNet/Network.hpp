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
        void setup(const Optimizer &optimizer, int epochs = 10, LOSS loss = LOSS::QUADRATIC);
        void addLayer(Layer &layer);
        void setBatchSize(int batchSize);
        void setLoss(LOSS loss);
        Layer getLayer(int index) const;
        Layer getOutputLayer() const;
        int getNumLayers() const;
        double train(std::vector<std::vector<double>> inputs, std::vector<double> labels);
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

        static double computeLoss(const MatrixXd &o, const Labels &y);
        static MatrixXd computeGradient(const MatrixXd &o, const Labels &y);
    };
}
