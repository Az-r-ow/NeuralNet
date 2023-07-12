#include "Neuron.hpp"

double Neuron::getWeight(int index) const {
    return this->m_weights.at(index);
}

double Neuron::getOutput() const {
    return this->output;
}

Neuron::Neuron(double bias): bias(bias) {}


float randomFloatInRange(int min, int max) {
   return min + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(max-min)));
}

/**
 * This comment should be removed after read 
 * I moved delete functions to the bottom because they're the
 * least interacted with 
*/
Neuron::~Neuron() {}