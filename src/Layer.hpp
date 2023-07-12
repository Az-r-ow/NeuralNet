#pragma once

#include <vector>
#include <cstdlib>
#include <string>
#include <Neuron.hpp>

using std::string;

class Layer
{
public:
    Layer(int nNeurons, double bias);
    int getNumNeurons() const;

    /**
     * Check comment in Neuron.cpp about the delete function
     */
    ~Layer();

private:
    std::vector<double> biases;
    std::vector<double> weights;
    std::vector<Neuron> neurons;
};
