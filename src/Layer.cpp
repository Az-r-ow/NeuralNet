#include "Layer.hpp"

Layer::Layer(int nNeurons, Activation activation, WeightInit weightInit, int bias)
{
    VectorXd outputs(nNeurons);
    this->outputs = outputs;
    this->bias = bias;
    this->weightInit = weightInit;
    this->setActivation(activation);
}

void Layer::initWeights(int numCols)
{
    double mean = 0.0, stddev = 0.0;
    MatrixXd weights(this->getNumNeurons(), numCols);
    this->weights = weights;

    // calculate mean and stddev based on init algo
    switch (this->weightInit)
    {
    case GLOROT:
        // sqrt(fan_avg)
        stddev = sqrt(static_cast<double>((numCols + this->getNumNeurons()) / 2));
        break;
    case HE:
        // sqrt(2/fan_in)
        stddev = sqrt(static_cast<double>(2 / numCols));
        break;
    case LACUN:
        // sqrt(1/fan_in)
        stddev = sqrt(static_cast<double>(1 / numCols));
        break;
    default:
        break;
    }

    // Init the weights
    this->weightInit == RANDOM ? randomWeightInit(&(this->weights)) : randomDistWeightInit(&(this->weights), mean, stddev);
}

void Layer::setActivation(Activation activation)
{
    switch (activation)
    {
    case RELU:
        this->activate = relu;
        break;
    case SIGMOID:
        this->activate = sigmoid;
        break;
    }

    return;
}

void Layer::feedInputs(vector<double> inputs)
{
    assert(inputs.size() == this->getNumNeurons());
    this->feedInputs(Matrix1d::Map(&inputs[0], 1, inputs.size()));
    return;
}

void Layer::feedInputs(Matrix1d inputs)
{
    assert(inputs.cols() == this->getNumNeurons());
    this->computeOutputs(inputs);
    return;
}

void Layer::printOutputs()
{
    std::cout << this->outputs << std::endl;
}

int Layer::getNumNeurons() const
{
    return this->outputs.cols();
}

void Layer::printWeights()
{
    std::cout << this->weights << std::endl;
    return;
}

/**
 * PRIVATE METHODS
 */

void Layer::computeOutputs(Matrix1d inputs)
{
    // Weighted sum
    Matrix1d wSum = inputs * this->weights;

    // Activate wSums
    for (int i = 0; i < wSum.cols(); i++)
        wSum[i] = this->activate(wSum[i]);

    this->setOutputs(Matrix1d::Map(&wSum[0], 1, wSum.cols()));
    return;
}

void Layer::setOutputs(Matrix1d outputs)
{
    this->outputs = outputs;
    return;
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