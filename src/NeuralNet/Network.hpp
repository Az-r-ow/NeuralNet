#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include "Layer.hpp"
#include "utils/Formatters.hpp"
#include "utils/Gauge.hpp"

namespace NeuralNet
{
    class Network
    {
    public:
        Network(double alpha = 0.001);
        void addLayer(Layer &layer);
        Layer getLayer(int index) const;
        Layer getOutputLayer() const;
        int getNumLayers() const;
        double train(std::vector<std::vector<double>> inputs, std::vector<double> labels);
        std::vector<std::vector<double>> predict(std::vector<std::vector<double>> inputs);
        ~Network();

    private:
        std::vector<Layer> layers;
        int cp = 0, tp = 0; // Correct Predictions, Total Predictions
        double alpha;       // Learning rate
        double loss = 1;

        std::vector<double> forwardProp(std::vector<double> inputs);
        double backProp(Labels y);
        double computeAccuracy(int predicted, int label);

        static double computeLoss(const MatrixXd &o, const Labels &y);
        static MatrixXd computeLossDer(const MatrixXd &o, const Labels &y);
    };
}
