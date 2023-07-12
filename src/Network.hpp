#pragma once

#include <vector>
#include <cstdlib>
#include <Layer.hpp>

using std::vector;
class Network
{
public:
    Network();
    void addLayer(Layer layer);
    int getNumLayer() const;
    void fit(vector<double> inputs, vector<double> labels);
    void predict(vector<double> outputs);

    /**
     * Check comment in Neuron.cpp about the delete function
     */
    ~Network();

private:
    Layer inputLayer;
    std::vector<Layer> layers;
    double learningRate;
    int neuronPerLayer;

    /* Private Methods */
    void forwardPropagate(string forwardFunctionName);
    void backPropagate(string lossFunctionName);
};
