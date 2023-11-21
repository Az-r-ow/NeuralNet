#include "Network.hpp"

using namespace NeuralNet;

Network::Network(double alpha) : alpha(alpha){};

size_t Network::getNumLayers() const
{
    return this->layers.size();
}

void Network::setup(const std::shared_ptr<Optimizer> &optimizer, int epochs, LOSS loss)
{
    this->optimizer = optimizer;
    this->epochs = epochs;
    this->lossFunc = loss;
    this->setLoss(loss);
    this->updateOptimizerSetup(this->layers.size());
}

void Network::addLayer(std::shared_ptr<Layer> &layer)
{
    size_t numLayers = this->layers.size();
    // Init layer with right amount of weights
    if (numLayers > 0)
    {
        int prevLayerNN = this->layers[this->layers.size() - 1]->getNumNeurons();
        layer->init(prevLayerNN);
    }

    this->layers.push_back(layer);
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

std::shared_ptr<Layer> Network::getLayer(int index) const
{
    assert(index < this->layers.size() && index >= 0);
    return this->layers.at(index);
}

std::shared_ptr<Layer> Network::getOutputLayer() const
{
    assert(this->layers.size() > 0);
    return this->layers[this->layers.size() - 1];
}

double Network::train(std::vector<std::vector<double>> inputs, std::vector<double> labels)
{
    return trainingProcess(inputs, labels);
}

double Network::train(std::vector<std::vector<std::vector<double>>> inputs, std::vector<double> labels)
{
    return trainingProcess(inputs, labels);
}

// Specific implementation of train that takes TrainingData class as input
double Network::train(TrainingData<std::vector<std::vector<double>>, std::vector<double>> trainingData)
{
    return this->trainer(trainingData);
}

double Network::train(TrainingData<std::vector<std::vector<std::vector<double>>>, std::vector<double>> trainingData)
{
    return this->trainer(trainingData);
}

template <typename D1, typename D2>
double Network::trainer(TrainingData<D1, D2> trainingData)
{
    if (trainingData.batched)
        return this->miniBatchTraining(trainingData);
    return this->batchTraining(trainingData);
}

template <typename D1, typename D2>
double Network::miniBatchTraining(TrainingData<D1, D2> trainingData)
{
    double loss;

    for (int e = 0; e < epochs; e++)
    {
        TrainingGauge g(trainingData.inputs.size(), 0, epochs, (e + 1));
        for (int b = 0; b < trainingData.inputs.size(); b++)
        {
            const int numOutputs = this->getOutputLayer()->getNumNeurons();
            const int inputsSize = trainingData.inputs.batches[b].size();
            Eigen::MatrixXd y = formatLabels(trainingData.labels.batches[b], {inputsSize, numOutputs});
            Eigen::MatrixXd grad = zeroMatrix({inputsSize, numOutputs});

            // computing outputs from forward propagation
            Eigen::MatrixXd o = this->forwardProp(trainingData.inputs.batches[b]);
            loss = this->cmpLoss(o, y) / inputsSize;
            grad = this->cmpGradient(o, y);
            this->backProp(grad);
            g.printWithLoss(loss);
        }
    }

    return loss;
}

template <typename D1, typename D2>
double Network::batchTraining(TrainingData<D1, D2> trainingData)
{
    double loss;
    const int numOutputs = this->getOutputLayer()->getNumNeurons();
    const int inputsSize = trainingData.inputs.data.size();
    Eigen::MatrixXd y = formatLabels(trainingData.labels.data, {inputsSize, numOutputs});
    Eigen::MatrixXd grad = zeroMatrix({inputsSize, numOutputs});

    for (int e = 0; e < epochs; e++)
    {
        Eigen::MatrixXd o = this->forwardProp(trainingData.inputs.data);

        loss = this->cmpLoss(o, y);

        grad = this->cmpGradient(o, y);

        this->backProp(grad);

        grad = zeroMatrix({inputsSize, numOutputs});
    }

    return loss;
}

template <typename D1, typename D2>
double Network::trainingProcess(std::vector<D1> inputs, std::vector<D2> labels)
{
    double loss;
    const int numOutputs = this->getOutputLayer()->getNumNeurons();
    const int inputsSize = inputs.size();
    Eigen::MatrixXd y = formatLabels(labels, {inputsSize, numOutputs});
    Eigen::MatrixXd grad = zeroMatrix({inputsSize, numOutputs});

    for (int e = 0; e < epochs; e++)
    {
        TrainingGauge progBar(inputsSize, 0, epochs, (e + 1));

        Eigen::MatrixXd o = this->forwardProp(inputs);

        loss = this->cmpLoss(o, y);

        grad = this->cmpGradient(o, y);

        this->backProp(grad);

        grad = zeroMatrix({inputsSize, numOutputs});
    }

    return loss;
}

Eigen::MatrixXd Network::predict(std::vector<std::vector<double>> inputs)
{
    std::vector<double> predictions(inputs.size());
    Eigen::MatrixXd mInputs = vectorToMatrixXd(inputs);

    Eigen::MatrixXd mPredictions = forwardProp(mInputs);
    return mPredictions;
}

Eigen::MatrixXd Network::predict(std::vector<std::vector<std::vector<double>>> inputs)
{
    Eigen::MatrixXd mPredictions = forwardProp(inputs);
    return mPredictions;
}

/**
 * Forward propagation
 */
Eigen::MatrixXd Network::forwardProp(std::vector<std::vector<std::vector<double>>> inputs)
{
    std::shared_ptr<Layer> firstLayer = this->layers[0];

    // Passing the inputs as outputs to the input layer
    firstLayer->feedInputs(inputs);

    Eigen::MatrixXd prevLayerOutputs = this->layers[0]->getOutputs();

    // Feeding the rest of the layers with the results of (L - 1)
    for (size_t l = 1; l < this->layers.size(); l++)
    {
        this->layers[l]->feedInputs(prevLayerOutputs);
        prevLayerOutputs = this->layers[l]->getOutputs();
    }

    return prevLayerOutputs;
}

Eigen::MatrixXd Network::forwardProp(std::vector<std::vector<double>> inputs)
{
    // Previous layer outputs
    Eigen::MatrixXd prevLayerO = vectorToMatrixXd(inputs);

    for (std::shared_ptr<Layer> &layer : layers)
    {
        layer->feedInputs(prevLayerO);
        prevLayerO = layer->getOutputs();
    }

    return prevLayerO;
}

Eigen::MatrixXd Network::forwardProp(Eigen::MatrixXd inputs)
{
    // Previous layer outputs
    Eigen::MatrixXd prevLayerO = inputs;

    for (std::shared_ptr<Layer> &layer : layers)
    {
        layer->feedInputs(prevLayerO);
        prevLayerO = layer->getOutputs();
    }

    return prevLayerO;
}

void Network::backProp(Eigen::MatrixXd grad)
{
    // Next Layer activation der dL/da(l - 1)
    Eigen::MatrixXd nextLayerADer = grad.transpose();

    for (size_t i = this->layers.size(); --i > 0;)
    {
        std::shared_ptr<Layer> cLayer = this->layers[i];
        std::shared_ptr<Layer> nLayer = this->layers[i - 1];

        // a'(L)
        Eigen::MatrixXd aDer = cLayer->diff(cLayer->outputs);

        // a(L - 1) . a'(L)
        Eigen::MatrixXd aDerNextDotaDer = nextLayerADer.array() * aDer.transpose().array();

        // dL/dw
        Eigen::MatrixXd wDer = aDerNextDotaDer * nLayer->getOutputs();

        // dL/db
        Eigen::MatrixXd bDer = aDerNextDotaDer.rowwise().sum().transpose();

        // dL/dA(l - 1)
        nextLayerADer = cLayer->weights * aDerNextDotaDer;

        // updating weights and biases
        this->optimizer->updateWeights(cLayer->weights, wDer.transpose());
        this->optimizer->updateBiases(cLayer->biases, bDer);
    }
}

void Network::updateOptimizerSetup(size_t numLayers)
{
    /**
     * This is a way to let adams know about the number of layers
     * With that it can setup the `l` variable and the std::vectors
     *
     * I'm not very proud of this method but so far it seems like the most convenient way
     */
    this->optimizer->insiderInit(numLayers);
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