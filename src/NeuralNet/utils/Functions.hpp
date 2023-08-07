#pragma once

#include <cmath>
#include <iostream>
#include <random>

namespace NeuralNet
{
  /**
   * Rand double generator that uses the Mersenne Twister algo
   */
  extern double mtRand(double min, double max)
  {
    std::random_device rseed;
    std::mt19937_64 rng(rseed());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(rng);
  };

  /* Activation Functions */
  extern double relu(double x)
  {
    return x < 0 ? 0 : x;
  };

  extern double sigmoid(double x)
  {
    return 1 / (1 + std::exp(-x));
  };

  /* Mathematical functions */
  /**
   * Returns the square of x
   */
  extern double sqr(double x)
  {
    return x * x;
  };
}