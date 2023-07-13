#include "Functions.hpp"

double mt_rand(double min, double max)
{
  std::random_device rseed;
  std::mt19937_64 rng(rseed());
  std::uniform_real_distribution<double> dist(min, max);

  return dist(rng);
};