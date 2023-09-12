#include "Network.hpp"

using namespace NeuralNet;

Network::Network(double alpha, int epochs, LOSS loss)
{
    this->alpha = alpha;
    this->epochs = epochs;
    this->setLoss(loss);
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

void Network::setBatchSize(int batchSize)
{
    this->batchSize = batchSize;
}

void Network::setLoss(LOSS loss)
{
    switch (loss)
    {
    case LOSS::QUADRATIC:
        this->cmpLoss = Quadratic::cmpLoss;
        this->cmpGradient = Quadratic::cmpGradient;
        break;
    case LOSS::MCE:
        this->cmpLoss = MCE::cmpLoss;
        this->cmpGradient = MCE::cmpGradient;
        break;
    default:
        assert(false && "Loss not defined");
        break;
    }
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

double Network::train(std::vector<std::vector<double>> inputs, std::vector<double> labels)
{
    int batchCount = 0;
    double loss;
    int numOutputs = this->getOutputLayer().getNumNeurons();
    int inputsSize = inputs.size();
    MatrixXd grad;

    for (int e = 0; e < this->epochs; e++)
    {
        TrainingGauge progBar(inputsSize, 0, this->epochs, (e + 1));
        for (size_t i = 0; i < inputsSize; i++)
        {
            std::vector<double> pred = forwardProp(inputs[i]);
            double accuracy = this->computeAccuracy(findIndexOfMax(pred), labels[i]);
            MatrixXd o = this->getOutputLayer().getOutputs();
            Labels y = formatLabels(labels[i], numOutputs);
            double loss = this->cmpLoss(o, y);
            grad = grad.rows() == 0 ? this->cmpGradient(o, y) : (grad.array() - this->cmpGradient(o, y).array()) / 2;
            progBar.printWithLAndA(loss, accuracy);

            if (i % this->batchSize == 0)
            {
                backProp(grad);
            }
        }
    }

    return loss;
}

std::vector<std::vector<double>> Network::predict(std::vector<std::vector<double>> inputs)
{
    std::vector<std::vector<double>> predictions;

    // Reserving space in anticipation
    reserve2d(predictions, inputs.size(), this->getOutputLayer().getNumNeurons());

    for (int i = 0; i < inputs.size(); i++)
    {
        std::vector<double> prediction = forwardProp(inputs[i]);
        predictions[i] = prediction;
    }

    return predictions;
}

/**
 * Forward propagation
 */
std::vector<double> Network::forwardProp(std::vector<double> inputs)
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

void Network::backProp(MatrixXd grad)
{
    // Next Layer activation der dL/da(l - 1)
    MatrixXd nextLayerADer = grad;

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
}

/**
 * Updates the correct prediction and the total prediction
 * Returns the accuracy of the model
 */
double Network::computeAccuracy(const int predicted, const int label)
{
    this->tp += 1;

    if (predicted == label)
    {
        this->cp += 1;
    }

    return static_cast<double>(cp) / tp;
}

/**
 * Quadratic  loss
 * todo : add more loss functions
 * - Cross-entropy
 * - Exponential
 * - Hellinger distance
 */
double Network::computeLoss(const MatrixXd &outputs, const Labels &y)
{
    MatrixXd cMatrix = outputs.array() - y;
    cMatrix = cMatrix.unaryExpr(&sqr);

    return cMatrix.sum();
}

MatrixXd Network::computeGradient(const MatrixXd &yHat, const Labels &y)
{
    assert(yHat.rows() == y.rows());
    return (yHat.array() - y.array()).matrix() * 2;
}

Network::~Network()
{
}