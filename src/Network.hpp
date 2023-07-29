#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include "Layer.hpp"

using std::vector;

class Network
{
public:
    Network(double learnRate);
    void addLayer(Layer &layer);
    Layer getLayer(int index);
    int getNumLayer() const;
    void train(vector<vector<double>> inputs, vector<double> labels);
    void predict(vector<double> outputs);
    ~Network();

private:
    vector<Layer> layers;
    double learnRate;
    int neuronPerLayer;
    double loss = 1;

    /* Private Methods */
    void forwardProp(vector<double> inputs);
    void backProp();

    static double loss(Matrix1d &o, Matrix1d &y);
};
