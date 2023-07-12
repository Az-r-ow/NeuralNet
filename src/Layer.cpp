#include "Layer.hpp"

Layer::Layer(int nNeurons, double bias)
{
    for (int i = 0; i < nNeurons; i++)
    {
        this->neurons.push_back(Neuron(bias));
    }
}

int Layer::getNumNeurons() const
{
    return this->neurons.size();
}

Layer::~Layer() {}