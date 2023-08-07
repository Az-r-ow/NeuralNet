#pragma once

#include <cmath>
#include <iostream>
#include <random>

namespace NeuralNet
{
  /**
   * Rand double generator that uses the Mersenne Twister algo
   */
  inline double mtRand(double min, double max)
  {
    std::random_device rseed;
    std::mt19937_64 rng(rseed());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(rng);
  };

  /* Activation Functions */
  inline double relu(double x)
  {
    return x < 0 ? 0 : x;
  };

  inline double sigmoid(double x)
  {
    return 1 / (1 + std::exp(-x));
  };

  /* Mathematical functions */
  /**
   * Returns the square of x
   */
  inline double sqr(double x)
  {
    return x * x;
  };
}