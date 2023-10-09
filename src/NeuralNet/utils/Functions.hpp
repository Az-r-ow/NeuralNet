#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <cstddef>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>

namespace fs = std::filesystem;

namespace NeuralNet
{

  /**
   * @brief Rand double generator that uses the Mersenne Twister algo
   *
   * @param min Minimum value generated
   * @param max Maximum value generated
   *
   * @return A random number between min and max
   */
  inline double mtRand(double min, double max)
  {
    assert(min < max);
    std::random_device rseed;
    std::mt19937_64 rng(rseed());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(rng);
  };

  /**
   * @brief Generates a vector with random doubles based on the previous mtRand function
   *
   * @param size The desired size of the resulting vector
   * @param min Minimum possible number (default : -10)
   * @param max Maximum possible number (default: 10)
   *
   * @return A vector with random doubles generated with the Mersenne Twister algo
   */
  inline std::vector<double> randDVector(int size, double min = -10, double max = 10)
  {
    std::vector<double> v;

    for (int i = 0; i < size; i++)
    {
      v.push_back(mtRand(min, max));
    }

    return v;
  }

  /**
   * @brief This function checks if a file exists and has a specific extension
   *
   * @param filePath The path of the file
   * @param extension The extension that's checked
   *
   * @return Returns true if the file exists and has the specified extension otherwise returns false
   */
  inline bool fileExistsWithExtension(const std::string &filePath, const std::string &extension)
  {
    // Check if the file exist
    if (fs::exists(filePath))
    {
      // Check if the file has specified extension
      fs::path file(filePath);
      return file.has_extension() && file.extension() == extension;
    }

    return false;
  }

  /* Mathematical functions */
  /**
   * @brief Function that calculates the square of a number
   *
   * @param x the number
   *
   * @return The square of x
   */
  inline double sqr(const double x)
  {
    return x * x;
  };

  /**
   * @brief 2d std::vector memory allocation function
   *
   * @param v the vector that needs size allocation
   * @param rows the number of rows to allocate
   * @param cols the number of cols to allocate
   *
   * This function just simplifies reserving space for a 2 dimensional vector
   * It's necessary if we know the size in advance because it can save a lot of unnecessary computations
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
   * @brief Find the row index of the max element in a Matrix
   *
   * @param m The Eigen::Matrix
   *
   * @return -1 if an error occurs or not found otherwise returns the row index of the element.
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

  /**
   * @brief This function takes a 2d vector and flattens it into a 1d vector
   *
   * @param input The 2D vector
   * @param rows The number of vectors
   * @param cols The number of cols inside each vector
   *
   * Passing the number of rows and columns makes this function much more efficient
   *
   * @return The resulting 1D vector
   */
  template <typename T>
  std::vector<T> flatten2DVector(const std::vector<std::vector<T>> &input, size_t rows, size_t cols)
  {
    // Asserting that the inputs respect the declared size
    assert(input.size() == rows);
    for (const std::vector<T> &row : input)
    {
      assert(row.size() == cols);
    }

    std::vector<T> result;
    result.reserve(rows * cols);

    // Flatten the 2D vector
    for (const std::vector<T> &row : input)
    {
      result.insert(result.end(), row.begin(), row.end());
    }

    return result;
  }
} // namespace NeuralNet