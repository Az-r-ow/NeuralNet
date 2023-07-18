#include "Layer.hpp"

Layer::Layer(int nNeurons, Activation activation, WeightInit weightInit, int bias)
{
    this->numNeurons = nNeurons;
    this->bias = bias;
    this->weightInit = weightInit;
}

void Layer::initWeights(int numRows)
{
    double mean = 0.0, stddev = 0.0;
    MatrixXd weights(numRows, this->numNeurons);
    this->weights = weights;

    // calculate mean and stddev based on init algo
    switch (this->weightInit)
    {
    case GLOROT:
        // sqrt(fan_avg)
        stddev = sqrt(static_cast<double>((numRows + this->numNeurons) / 2));
        break;
    case HE:
        // sqrt(2/fan_in)
        stddev = sqrt(static_cast<double>(2 / numRows));
        break;
    case LACUN:
        // sqrt(1/fan_in)
        stddev = sqrt(static_cast<double>(1 / numRows));
        break;
    default:
        break;
    }

    // Init the weights
    this->weightInit == RANDOM ? randomWeightInit(&(this->weights)) : randomDistWeightInit(&(this->weights), mean, stddev);
}

int Layer::getNumNeurons() const
{
    return this->numNeurons;
}

void Layer::printWeights()
{
    std::cout << this->weights << std::endl;
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