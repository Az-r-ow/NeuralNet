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

Layer Network::getLayer(int index) const
{
    assert(index < this->layers.size() && index >= 0);
    return this->layers.at(index);
}

Layer Network::getOutputLayer() const
{
    assert(this->layers.size() > 0);
    return this->layers[this->layers.size() - 1];
}

double Network::train(vector<vector<double>> inputs, vector<double> labels)
{
    double loss;
    int numOutputs = this->getOutputLayer().getNumNeurons();
    int inputsSize = inputs.size();
    TrainingGauge progBar("Training : ", inputsSize);

    for (int i = 0; i < inputsSize; i++)
    {
        forwardProp(inputs[i]);
        Labels y = formatLabels(labels[i], numOutputs);
        loss = backProp(y);
        progBar.printWithError(loss);
    }

    return loss;
}

vector<vector<double>> predict(vector<vector<double>> inputs)
{
    vector<vector<double>> predictions;

    // Reserving space in anticipation
    reserve2d(predictions, inputs.size(), this->getOutputLayer().getNumNeurons());

    for (int i = 0; i < inputs.size(); i++)
    {
        vector<double> prediction = forwardProp(inputs[i]);
        predictions[i] = prediction;
    }

    return predictions;
}

/**
 * Forward propagation
 */
vector<double> Network::forwardProp(vector<double> inputs)
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

    return formatOutputs(this->getOutputLayer().getOutputs());
}

double Network::backProp(Labels y)
{
    // 1 - compute the lossDer
    MatrixXd oLayerOutputs = this->getOutputLayer().getOutputs();

    // Next Layer activation der dL/da(l - 1)
    MatrixXd nextLayerADer = computeLossDer(oLayerOutputs, y);

    for (unsigned i = this->layers.size(); --i > 0;)
    {
        Layer &cLayer = this->layers[i];
        Layer &nLayer = this->layers[i - 1];

        // a'(L)
        MatrixXd aDer = cLayer.diff(cLayer.outputs);
        // a(L - 1) . a'(L)
        MatrixXd aDerNextDotaDer = nextLayerADer.array() * aDer.array();

        // dL/dw
        MatrixXd wDer = aDerNextDotaDer * nLayer.getOutputs().transpose();
        // dL/db
        MatrixXd bDer = aDerNextDotaDer;
        // dL/dA(l - 1)
        nextLayerADer = cLayer.weights * aDerNextDotaDer;

        // updating weights and biases
        cLayer.weights = cLayer.weights.array() - (this->alpha * wDer.transpose()).array();
        cLayer.biases = cLayer.biases.array() - (this->alpha * bDer.transpose()).array();
    }

    return computeLoss(oLayerOutputs, y);
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
    cMatrix = cMatrix.unaryExpr(&sqr);

    return cMatrix.sum();
}

MatrixXd Network::computeLossDer(MatrixXd &yHat, Labels &y)
{
    assert(yHat.rows() == y.rows());
    return (yHat.array() - y.array()).matrix() * 2;
}

Network::~Network()
{
}