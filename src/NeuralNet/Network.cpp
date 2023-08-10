#include "Network.hpp"

using namespace NeuralNet;

Network::Network(double alpha)
{
    this->alpha = alpha;
}

int Network::getNumLayers() const
{
    return this->layers.size(); 
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
    assert(index < this->layers.size() && index >= 0);
    return this->layers.at(index);
}

Layer Network::getOutputLayer()
{
    assert(this->layers.size() > 0);
    return this->layers[this->layers.size() - 1];
}

void Network::train(vector<vector<double>> inputs, vector<double> labels)
{
    int numOutputs = this->getOutputLayer().getNumNeurons();
    for (int i = 0; i < inputs.size(); i++)
    {
        forwardProp(inputs[i]);
        Labels y = this->formatLabels(labels[i], numOutputs);
        backProp(y);
    }
}

void Network::forwardProp(vector<double> inputs)
{

    bool first = true;

    Layer &firstLayer = this->layers[0];

    // Passing the inputs as outputs to the input layer
    firstLayer.setOutputs(inputs);

    MatrixXd prevLayerOutputs = this->layers[0].getOutputs();

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
    MatrixXd oLayerOutputs = this->getOutputLayer().getOutputs();

    // Next Layer activation der dL/da(l - 1)
    MatrixXd nextLayerADer = computeLossDer(oLayerOutputs, y);

    for (unsigned i = this->layers.size(); --i > 0;)
    {
        Layer &cLayer = this->layers[i];
        Layer &nLayer = this->layers[i - 1];

        MatrixXd sigDer = computeSigmoidDer(cLayer.outputs);
        MatrixXd aDerDotSigDer = nextLayerADer.array() * sigDer.array();

        // dL/dw
        MatrixXd wDer = aDerDotSigDer * nLayer.getOutputs().transpose();
        // dL/db
        MatrixXd bDer = aDerDotSigDer;
        // dL/dA(l - 1)
        nextLayerADer = cLayer.weights * aDerDotSigDer;

        // updating weights and biases
        cLayer.weights = cLayer.weights.array() - (this->alpha * wDer.transpose()).array();
        cLayer.biases = cLayer.biases.array() - (this->alpha * bDer.transpose()).array();
    }

    return;
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
    cMatrix.unaryExpr(&sqr);
    return cMatrix.sum();
}

MatrixXd Network::computeLossDer(MatrixXd &yHat, Labels &y)
{
    assert(yHat.rows() == y.rows());
    return (yHat.array() - y.array()).matrix() * 2;
}

MatrixXd Network::computeSigmoidDer(MatrixXd &a)
{
    return a.array() * (1 - a.array());
}

Labels Network::formatLabels(int label, int rows)
{
    assert(label <= rows && label > 0);
    // Init 0 value Matrix (outputNeurons, 1)
    Labels labels = MatrixXd::Zero(rows, 1);
    labels(label - 1, 0) = 1;

    return labels;
}

Network::~Network()
{
}