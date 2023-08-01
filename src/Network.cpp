#include "Network.hpp"

Network::Network(double learnRate)
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
    int numOutputs = this->getOutputLayer().getNumNeurons();
    for (int i = 0; i < inputs.size(); i++)
    {
        forwardProp(inputs[i]);
        Labels y = Eigen::ArrayXd::Zero(numOutputs);
        y(labels[i], 0) = 1;
        backProp(y);
    }
}

Layer &Network::getOutputLayer()
{
    return this->layers[this->layers.size() - 1];
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

void Network::backProp(Labels y)
{
    // 1 - compute the lossDer
    // 2 - compute the output layer's activation derivative
    // 3 - compute the weight's derivative

    // 4 - Propagate using the activation layer's derivative
}

/**
 * Quadratic  loss
 * todo : add more loss functions
 * - Cross-entropy
 * - Exponential
 * - Hellinger distance
 */
double Network::computeLoss(MatrixXd &outputs, Labels &y)
{
    MatrixXd cMatrix = outputs.array() - y;
    cMatrix.unaryExpr(&Sqr);
    return cMatrix.sum();
}

MatrixXd Network::computeLossDer(MatrixXd &outputs, Labels &y)
{
    assert(outputs.rows() != y.rows());

    return 2 * (outputs.array() - y.array());
}

Network::~Network()
{
}