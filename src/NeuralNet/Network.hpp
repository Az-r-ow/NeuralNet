#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include "Layer.hpp"
#include "utils/Formatters.hpp"

namespace NeuralNet
{
    class Network
    {
    public:
        Network(double alpha = 0.1);
        void addLayer(Layer &layer);
        Layer getLayer(int index);
        int getNumLayers() const;
        void train(vector<vector<double>> inputs, vector<double> labels);
        void predict(vector<double> outputs);
        ~Network();

    private:
        vector<Layer> layers;
        double alpha; // Learning rate
        int neuronPerLayer;
        double loss = 1;

        Layer getOutputLayer();
        void forwardProp(vector<double> inputs);
        void backProp(Labels y);

        static double computeLoss(MatrixXd &o, Labels &y);
        static MatrixXd computeLossDer(MatrixXd &o, Labels &y);
        static MatrixXd computeSigmoidDer(MatrixXd &a);
    };
}
