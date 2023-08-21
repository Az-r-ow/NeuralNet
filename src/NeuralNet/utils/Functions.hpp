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

  inline std::vector<double> randDVector(int size, double min = -10, double max = 10)
  {
    std::vector<double> v;

    for (int i = 0; i < size; i++)
    {
      v.push_back(mtRand(min, max));
    }

    return v;
  }

  /* Mathematical functions */
  /**
   * Returns the square of x
   */
  inline double sqr(double x)
  {
    return x * x;
  };
}