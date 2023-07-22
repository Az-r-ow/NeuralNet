#pragma once

#include <vector>
#include <cstdlib>
#include "Layer.hpp"

using std::vector;

class Network
{
public:
    Network();
    void addLayer(Layer layer);
    Layer getLayer(int index);
    int getNumLayer() const;
    void train(vector<vector<double>> inputs, vector<double> labels);
    void predict(vector<double> outputs);
    ~Network();

private:
    vector<Layer> layers;
    double learningRate;
    int neuronPerLayer;

    /* Private Methods */
    void forwardProp(vector<double> inputs);
};
