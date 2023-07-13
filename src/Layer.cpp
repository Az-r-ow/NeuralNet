#include "Layer.hpp"

Layer::Layer(int nNeurons, Activation activation, int bias)
{
    this->numNeurons = nNeurons;
    this->bias = bias;
}

void Layer::initWeights(int numRows)
{
    MatrixXd weights(numRows, this->numNeurons);

    // Init the weights
    for (int col = 0; col < this->numNeurons; col++)
    {
        for (int row = 0; row < numRows; row++)
        {
            weights(row, col) = mt_rand(-1, 1);
        }
    }

    this->weights = weights;
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

Layer::~Layer() {}