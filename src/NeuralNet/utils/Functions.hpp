#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <cstddef>
#include <Eigen/Dense>

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
  inline double sqr(const double x)
  {
    return x * x;
  };

  /**
   * 2d std::vector memory allocation function
   */
  template <typename T>
  inline void reserve2d(std::vector<std::vector<T>> &v, int rows, int cols)
  {
    // reserve space for num rows
    v.reserve(rows);

    // reserve space for each row
    for (int i = 0; i < rows; i++)
    {
      v.push_back(std::vector<T>());
      v[i].reserve(cols);
    }
  };

  /**
   * Find the row index of the Max element in an array
   */
  inline int findRowIndexOfMaxEl(const Eigen::MatrixXd &m)
  {
    // Find the maximum value in the matrix
    double maxVal = m.maxCoeff();

    // Find the row index by iterating through rows
    for (int i = 0; i < m.rows(); ++i)
    {
      if ((m.row(i).array() == maxVal).any())
      {
        return i;
      }
    }

    // Return -1 if not found (this can be handled based on your use case)
    return -1;
  };
}