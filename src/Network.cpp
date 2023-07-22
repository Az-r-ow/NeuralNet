#include "Network.hpp"

Network::Network() {}

int Network::getNumLayer() const
{
    return this->layers.size() + 1; // see if input layer included
}

void Network::addLayer(Layer layer)
{
    // Init layer with right amount of weights
    if (this->layers.size() > 0)
    {
        int prevLayerNumNeurons = this->layers.at(this->layers.size() - 1).getNumNeurons();
        layer.initWeights(prevLayerNumNeurons);
    }
    this->layers.push_back(layer);
}

Layer Network::getLayer(int index)
{
    return this->layers.at(index);
}

void Network::train(vector<vector<double>> inputs, vector<double> labels)
{
    for (vector<double> input : inputs)
    {
        // forwardProp(input);
    }
}

void Network::forwardProp(vector<double> input)
{
    this->layers.front().feedInputs(input);
}

Network::~Network()
{
}