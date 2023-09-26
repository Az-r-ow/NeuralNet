#include "Network.hpp"

using namespace NeuralNet;

Network::Network(double alpha) : alpha(alpha){};

int Network::getNumLayers() const
{
    return this->layers.size();
}

void Network::setup(const Optimizer &optimizer, int epochs, LOSS loss)
{
    this->optimizer = optimizer;
    this->epochs = epochs;
    this->setLoss(loss);
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
    double loss;
    const int numOutputs = this->getOutputLayer().getNumNeurons();
    const int inputsSize = inputs.size();
    MatrixXd grad = this->nullifyGradient();

    for (int e = 0; e < this->epochs; e++)
    {
        TrainingGauge progBar(inputsSize, 0, this->epochs, (e + 1));
        for (size_t i = 0; i < inputsSize; i++)
        {
            MatrixXd o = forwardProp(inputs[i]);

            double accuracy = this->computeAccuracy(findRowIndexOfMaxEl(o), labels[i]);
            Labels y = formatLabels(labels[i], numOutputs);
            loss = this->cmpLoss(o, y);

            // sum grads
            grad = grad.array() + this->cmpGradient(o, y).array();

            // Printing progress and results
            progBar.printWithLAndA(loss, accuracy);

            if ((i + 1) % this->batchSize == 0)
            {
                grad = grad.array() / this->batchSize;
                backProp(grad);

                // Reset gradient for next mini-batch
                grad = this->nullifyGradient();
            }
        }
    }

    return loss;
}

std::vector<double> Network::predict(std::vector<std::vector<double>> inputs)
{
    std::vector<double> predictions(inputs.size());

    for (int i = 0; i < inputs.size(); i++)
    {
        MatrixXd prediction = forwardProp(inputs[i]);
        predictions[i] = findRowIndexOfMaxEl(prediction);
    }

    return predictions;
}

/**
 * Forward propagation
 */
MatrixXd Network::forwardProp(std::vector<double> inputs)
{
    Layer &firstLayer = this->layers[0];

    // Passing the inputs as outputs to the input layer
    firstLayer.setOutputs(inputs);

    MatrixXd prevLayerOutputs = this->layers[0].getOutputs();

    // Feeding the rest of the layers with the results of (L - 1)
    for (size_t l = 1; l < this->layers.size(); l++)
    {
        this->layers[l].feedInputs(prevLayerOutputs);
        prevLayerOutputs = this->layers[l].getOutputs();
    }

    return prevLayerOutputs;
}

void Network::backProp(MatrixXd grad)
{
    // Next Layer activation der dL/da(l - 1)
    MatrixXd nextLayerADer = grad;

    for (size_t i = this->layers.size(); --i > 0;)
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
        this->optimizer.updateWeights(cLayer.weights, wDer);
        this->optimizer.updateBiases(cLayer.biases, bDer);
    }
}

/**
 * Reset the gradient to 0
 */
MatrixXd Network::nullifyGradient()
{
    int rows = this->getOutputLayer().getNumNeurons();

    return MatrixXd::Zero(rows, 1);
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

Network::~Network()
{
}