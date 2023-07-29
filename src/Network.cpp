#include "Network.hpp"

Network::Network(double learnRate = 0.1)
{
    this->learnRate = learnRate;
}

int Network::getNumLayer() const
{
    return this->layers.size() + 1; // see if input layer included
}

void Network::addLayer(Layer &layer)
{
    // Init layer with right amount of weights
    if (this->layers.size() > 0)
    {
        int prevLayerNN = this->layers[this->layers.size() - 1].getNumNeurons();
        layer.initWeights(prevLayerNN);
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
        forwardProp(input);
    }
}

void Network::forwardProp(vector<double> inputs)
{

    bool first = true;

    Layer &firstLayer = this->layers[0];

    // Passing the inputs to the first layer
    firstLayer.initWeights(inputs.size());
    firstLayer.feedInputs(inputs);

    Matrix1d prevLayerOutputs = this->layers[0].getOutputs();

    // Feeding the rest of the layers with the results of (L - 1)
    for (Layer &layer : this->layers)
    {
        // Skipping the first layer since already fed
        if (first)
        {
            first = false;
            continue;
        }

        layer.feedInputs(prevLayerOutputs);
        prevLayerOutputs = layer.getOutputs();
    }
}

void Network::backProp(Matrix1d y)
{
    Layer &outputLayer = this->layers[this->layers.size() - 1];

    // Get total loss
    this->loss = loss(outputLayer.outputs, y);

    for (unsigned i = this->layers.size(); i-- > 0;)
    {
        // propagate backwards
    }
}

/**
 * Quadratic  loss
 * todo : add more loss functions
 * - Cross-entropy
 * - Exponential
 * - Hellinger distance
 */
double Network::loss(Matrix1d &outputs, Matrix1d &y)
{
    Matrix cMatrix = outputs - y;
    cMatrix.unaryExpr(&Sqr);
    return cMatrix.sum();
}

Network::~Network()
{
}