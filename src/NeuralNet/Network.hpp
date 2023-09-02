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
        double train(vector<vector<double>> inputs, vector<double> labels);
        vector<double> predict(vector<vector<double>> inputs);
        ~Network();

    private:
        vector<Layer> layers;
        double alpha; // Learning rate
        double loss = 1;

        vector<double> forwardProp(vector<double> inputs);
        double backProp(Labels y);

        static double computeLoss(MatrixXd &o, Labels &y);
        static MatrixXd computeLossDer(MatrixXd &o, Labels &y);
        static MatrixXd computeSigmoidDer(MatrixXd &a);
    };
}
