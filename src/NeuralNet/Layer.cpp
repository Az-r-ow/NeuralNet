#include "Layer.hpp"

using namespace NeuralNet;

Layer::Layer(int nNeurons, ACTIVATION activationName, WEIGHT_INIT weightInit, int bias)
{
    this->outputs = MatrixXd::Zero(nNeurons, 1);
    this->biases = MatrixXd::Constant(1, nNeurons, bias);
    this->weightInit = weightInit;
    this->setActivation(activationName);
}

void Layer::initWeights(int numRows)
{
    double mean = 0.0, stddev = 0.0;
    int numCols = this->getNumNeurons();
    this->weights = MatrixXd::Zero(numRows, numCols);

    // calculate mean and stddev based on init algo
    switch (this->weightInit)
    {
    case WEIGHT_INIT::GLOROT:
        // sqrt(fan_avg)
        stddev = sqrt(static_cast<double>((numRows + numCols) / 2));
        break;
    case WEIGHT_INIT::HE:
        // sqrt(2/fan_in)
        stddev = sqrt(static_cast<double>(2 / numRows));
        break;
    case WEIGHT_INIT::LACUN:
        // sqrt(1/fan_in)
        stddev = sqrt(static_cast<double>(1 / numRows));
        break;
    default:
        break;
    }

    // Init the weights
    this->weightInit == WEIGHT_INIT::RANDOM ? randomWeightInit(&(this->weights), -1, 1) : randomDistWeightInit(&(this->weights), mean, stddev);
}

void Layer::setActivation(ACTIVATION activation)
{
    switch (activation)
    {
    case ACTIVATION::SIGMOID:
        this->activate = Sigmoid::activate;
        this->diff = Sigmoid::diff;
        break;
    case ACTIVATION::RELU:
        this->activate = Relu::activate;
        this->diff = Relu::diff;
        break;
    case ACTIVATION::SOFTMAX:
        this->activate = Softmax::activate;
        this->diff = Softmax::diff;
        break;
    /**
     * Add cases as I add activations
     */
    default:
        assert(false && "Activation not defined");
    }

    return;
}

void Layer::feedInputs(std::vector<double> inputs)
{
    assert(inputs.size() == this->weights.rows());
    this->feedInputs(MatrixXd::Map(&inputs[0], inputs.size(), 1));
    return;
}

void Layer::feedInputs(MatrixXd inputs)
{
    assert(inputs.rows() == this->weights.rows());
    this->computeOutputs(inputs);
    return;
}

int Layer::getNumNeurons() const
{
    return this->outputs.rows();
}

MatrixXd Layer::getOutputs()
{
    return this->outputs;
}

MatrixXd Layer::getWeights() const
{
    return this->weights;
}

void Layer::printWeights()
{
    std::cout << this->weights << std::endl;
    return;
}

void Layer::printOutputs()
{
    std::cout << this->outputs << std::endl;
    return;
}

/**
 * PRIVATE METHODS
 */

void Layer::computeOutputs(MatrixXd inputs)
{
    // Weighted sum
    MatrixXd wSum = inputs.transpose() * this->weights;
    wSum += this->biases;
    wSum = wSum.transpose();

    this->outputs = this->activate(wSum);
    return;
}

void Layer::setOutputs(std::vector<double> outputs)
{
    assert(outputs.size() == this->getNumNeurons());
    this->outputs = MatrixXd::Map(&outputs[0], this->getNumNeurons(), 1);
}

void Layer::randomWeightInit(MatrixXd *weights, double min, double max)
{
    for (int col = 0; col < weights->cols(); col++)
    {
        for (int row = 0; row < weights->rows(); row++)
        {
            weights->operator()(row, col) = mtRand(min, max);
        }
    }

    return;
}

void Layer::randomDistWeightInit(MatrixXd *weights, double mean, double stddev)
{
    std::random_device rseed;
    std::default_random_engine generator(rseed());
    std::normal_distribution<double> distribution(mean, stddev);

    for (int col = 0; col < weights->cols(); col++)
    {
        for (int row = 0; row < weights->rows(); row++)
        {
            weights->operator()(row, col) = distribution(generator);
        }
    }

    return;
}

Layer::~Layer() {}