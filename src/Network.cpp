#include "Network.hpp"

int Network::getNumLayer() const
{
    return this->layers.size() + 1; // see if input layer included
}

void Network::addLayer(Layer layer)
{
    this->layers.push_back(layer);
}

Network::~Network() {}