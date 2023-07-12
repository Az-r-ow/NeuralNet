#pragma once

#include <vector>
#include <cstdlib>

class Neuron {
    public:
        Neuron(double bias);
        ~Neuron();
        double getWeight(int index) const;
        double getOutput() const;
    private:
        std::vector<double> m_weights;
        double bias;
        double output;
};

float randomFloatInRange(int min, int max);
