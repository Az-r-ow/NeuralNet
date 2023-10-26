#pragma once

#include "interfaces/Layer.hpp"

namespace NeuralNet
{
    class Dense : public Layer
    {
    public:
        Dense(int nNeurons, ACTIVATION activation = ACTIVATION::SIGMOID, WEIGHT_INIT weightInit = WEIGHT_INIT::RANDOM, int bias = 0) : Layer(nNeurons, activation, weightInit, bias) {}

        ~Dense() {}
    };
}
