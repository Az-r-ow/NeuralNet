#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

namespace fs = std::filesystem;

namespace NeuralNet {

/**
 * @brief Rand double generator that uses the Mersenne Twister algo
 *
 * @param min Minimum value generated
 * @param max Maximum value generated
 *
 * @return A random number between min and max
 */
inline double mtRand(double min, double max) {
  assert(min < max);
  std::random_device rseed;
  std::mt19937_64 rng(rseed());
  std::uniform_real_distribution<double> dist(min, max);

  return dist(rng);
};

/**
 * @brief Generates a vector with random doubles based on the previous mtRand
 * function
 *
 * @param size The desired size of the resulting vector
 * @param min Minimum possible number (default : -10)
 * @param max Maximum possible number (default: 10)
 *
 * @return A vector with random doubles generated with the Mersenne Twister algo
 */
inline std::vector<double> randDVector(int size, double min = -10,
                                       double max = 10) {
  std::vector<double> v;

  for (int i = 0; i < size; i++) {
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
 * @return Returns true if the file exists and has the specified extension
 * otherwise returns false
 */
inline bool fileExistsWithExtension(const std::string &filePath,
                                    const std::string &extension) {
  // Check if the file exist
  if (fs::exists(filePath)) {
    // Check if the file has specified extension
    fs::path file(filePath);
    return file.has_extension() && file.extension() == extension;
  }

  return false;
}

/**
 * @brief This function checks if a file has a specific extension
 *
 * @param filePath The path of the file
 * @param extension The extension that's checked
 *
 * @return Returns true if the file has the specified extension otherwise
 * returns false
 */
inline bool fileHasExtension(const std::string &filePath,
                             const std::string &extension) {
  fs::path file(filePath);
  return file.has_extension() && file.extension() == extension;
}

/**
 * @brief Check if a folder exists or not based on its path
 *
 * @param folderPath The path to the folder
 *
 * @return Returns `true` if the folder exists
 */
inline bool folderExists(const std::string &folderPath) {
  return fs::exists(folderPath) && fs::is_directory(folderPath);
}

/**
 * @brief Generates a file path from folder path and fileName
 *
 * @param folderPath The path to the folder
 * @param fileName The name of the file
 *
 * @return Returns the filepath with the given arguments
 */
inline std::string constructFilePath(const std::string &folderPath,
                                     const std::string &fileName) {
  std::string filepath;

// Running on windows
#ifdef _WIN32
  if (!folderPath.empty() && folderPath.back() != "\\")
    filepath = folderPath + "\\";
  else
    filepath = folderPath;
#else
  // Not running on Windows
  if (!folderPath.empty() && folderPath.back() != '/')
    filepath = folderPath + "/";
  else
    filepath = folderPath;
#endif

  return filepath + fileName;
};

/* MATHEMATICAL FUNCTIONS */

/**
 * @brief Function that calculates the square of a number
 *
 * @param x the number
 *
 * @return The square of x
 */
inline constexpr double sqr(const double x) { return x * x; };

/* VECTOR OPERATIONS */

/**
 * @brief 2d std::vector memory allocation function
 *
 * @param v the vector that needs size allocation
 * @param rows the number of rows to allocate
 * @param cols the number of cols to allocate
 *
 * This function just simplifies reserving space for a 2 dimensional vector
 * It's necessary if we know the size in advance because it can save a lot of
 * unnecessary computations
 */
template <typename T>
inline void reserve2d(std::vector<std::vector<T>> &v, int rows, int cols) {
  // reserve space for num rows
  v.reserve(rows);

  // reserve space for each row
  for (int i = 0; i < rows; i++) {
    v.push_back(std::vector<T>());
    v[i].reserve(cols);
  }
};

/**
 * @brief Find the row index of the max element in a Matrix
 *
 * @param m The Eigen::Matrix
 *
 * @return -1 if an error occurs or not found otherwise returns the row index of
 * the element.
 */
inline int findRowIndexOfMaxEl(const Eigen::MatrixXd &m) {
  // Find the maximum value in the matrix
  double maxVal = m.maxCoeff();

  // Find the row index by iterating through rows
  for (int i = 0; i < m.rows(); ++i) {
    if ((m.row(i).array() == maxVal).any()) {
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
 * Passing the number of rows and columns makes this function much more
 * efficient
 *
 * @return The resulting 1D vector
 */
template <typename T>
inline std::vector<T> flatten2DVector(const std::vector<std::vector<T>> &input,
                                      size_t rows, size_t cols) {
  // Asserting that the inputs respect the declared size
  assert(input.size() == rows);
  for (const std::vector<T> &row : input) {
    assert(row.size() == cols);
  }

  std::vector<T> result;
  result.reserve(rows * cols);

  // Flatten the 2D vector
  for (const std::vector<T> &row : input) {
    result.insert(result.end(), row.begin(), row.end());
  }

  return result;
}

/**
 * @brief This function takes a vector and a value and returns the index of the
 * value in that vector.
 *
 * @param v The vector
 * @param el The element to look for in the vector
 *
 * @return The index of the element in the vector `-1` if not found.
 */
template <typename T>
inline int findIndexOf(const std::vector<T> &v, const T &el) {
  auto it = std::find(v.begin(), v.end(), el);

  if (it == v.end()) return -1;

  return it - v.begin();
}

/* MATRIX OPERATIONS */
inline Eigen::MatrixXd zeroMatrix(const std::tuple<int, int> size) {
  return Eigen::MatrixXd::Zero(std::get<0>(size), std::get<1>(size));
}

inline Eigen::MatrixXd vectorToMatrixXd(std::vector<std::vector<double>> &v) {
  if (v.empty() || v[0].empty()) return Eigen::MatrixXd(0, 0);

  int rows = v.size();
  int cols = v[0].size();

  // Flatten the vector of vectors into a single vector
  std::vector<double> flat;
  flat.reserve(rows * cols);
  for (const auto &row : v) {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      flat.data(), rows, cols);
};

/**
 * @brief initialises a matrix with random values that ranges between the
 * defined boundaries
 *
 * @param weightsMatrix a pointer to the weights matrix or any Eigen::MatrixXd
 * @param min The min value in the range (default: -1)
 * @param max The max value in the range (default: 1)
 *
 * @return void
 */
static void randomWeightInit(Eigen::MatrixXd *weightsMatrix, double min = -1.0,
                             double max = 1.0) {
  for (int col = 0; col < weightsMatrix->cols(); col++) {
    for (int row = 0; row < weightsMatrix->rows(); row++) {
      weightsMatrix->operator()(row, col) = mtRand(min, max);
    }
  }

  return;
};

/**
 * @brief Method that sets the value of a matrix to follow a certain random Dist
 *
 * @param weightsMatrix A pointer to a weight matrix or any Eigen::MatrixXd
 * @param mean The mean for the std dist
 * @param stddev The standard deviation
 *
 * @return void
 */
static void randomDistMatrixInit(Eigen::MatrixXd *weightsMatrix, double mean,
                                 double stddev) {
  std::random_device rseed;
  std::default_random_engine generator(rseed());
  std::normal_distribution<double> distribution(mean, stddev);

  for (int col = 0; col < weightsMatrix->cols(); col++) {
    for (int row = 0; row < weightsMatrix->rows(); row++) {
      weightsMatrix->operator()(row, col) = distribution(generator);
    }
  }

  return;
};

/**
 * @brief takes in a matrix and returns its hardmax equivalent.
 *
 * @param mat the inputs matrix
 *
 * @return the hardmax version of the matrix.
 */
static Eigen::MatrixXd hardmax(const Eigen::MatrixXd &mat) {
  Eigen::MatrixXd hardmaxMatrix = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());

  for (int i = 0; i < mat.rows(); ++i) {
    int maxIndex;
    mat.row(i).maxCoeff(&maxIndex);

    hardmaxMatrix(i, maxIndex) = 1;
  }

  return hardmaxMatrix;
}

/**
 * @deprecated This function will be removed/replaced soon
 * @brief round the number < to the given threshold to 0
 *
 * @param logits Matrix of doubles
 * @param threshold a double (default: 0.01)
 *
 * @return the same matrix with the values < threshold = 0
 */
static Eigen::MatrixXd trim(const Eigen::MatrixXd &logits,
                            double threshold = 0.01) {
  return (logits.array() < threshold).select(0, logits);
}

/**
 * @brief round the number < to the given threshold to the given threshold
 *
 * @param logits Matrix of doubles
 * @param threshold a double (default: 0.01)
 *
 * @return the same matrix with the values < threshold = threshold
 */
static Eigen::MatrixXd thresh(const Eigen::MatrixXd &logits,
                              double threshold = 0.01) {
  return (logits.array() < threshold).select(threshold, logits);
}

/* SIGNAL HANDLING */
static void signalHandler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";

  // cleanup and close up stuff here
  // terminate program
  exit(signum);
};

}  // namespace NeuralNet