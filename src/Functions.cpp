#include "Functions.hpp"

double mtRand(double min, double max)
{
  std::random_device rseed;
  std::mt19937_64 rng(rseed());
  std::uniform_real_distribution<double> dist(min, max);

  return dist(rng);
};

double relu(double x)
{
  return x < 0 ? 0 : x;
}

double sigmoid(double x)
{
  return 1 / (1 + std::exp(-x));
}

double sqr(double x)
{
  return x * x;
}
