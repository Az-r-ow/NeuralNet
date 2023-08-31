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
        void train(vector<vector<double>> inputs, vector<double> labels);
        void predict(vector<double> outputs);
        ~Network();

    private:
        vector<Layer> layers;
        double alpha; // Learning rate
        int neuronPerLayer;
        double loss = 1;

        void forwardProp(vector<double> inputs);
        double backProp(Labels y);

        static double computeLoss(MatrixXd &o, Labels &y);
        static MatrixXd computeLossDer(MatrixXd &o, Labels &y);
        static MatrixXd computeSigmoidDer(MatrixXd &a);
    };
}
