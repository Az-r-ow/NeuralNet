#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include "Layer.hpp"

class Network
{
public:
    Network(double alpha = 0.1);
    void addLayer(Layer &layer);
    Layer getLayer(int index);
    int getNumLayer() const;
    void train(vector<vector<double>> inputs, vector<double> labels);
    void predict(vector<double> outputs);
    ~Network();

private:
    vector<Layer> layers;
    double alpha; // Learning rate
    int neuronPerLayer;
    double loss = 1;

    /* Private Methods */
    Layer getOutputLayer();
    void forwardProp(vector<double> inputs);
    void backProp(Labels y);

    /* private static functions */
    static double computeLoss(MatrixXd &o, Labels &y);
    static MatrixXd computeLossDer(MatrixXd &o, Labels &y);
    static MatrixXd computeSigmoidDer(MatrixXd &a);
    static Labels formatLabels(int label, int rows);
};
