#pragma once

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <set>
#include <vector>

#define EPSILON 1e-3

void REQUIRE_MATRIX_VALUES_UNIQUE(Eigen::MatrixXd &matrix) {
  std::set<int> seenValues;

  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < matrix.cols(); j++) {
      REQUIRE(seenValues.find(matrix(i, j)) == seenValues.end());
      seenValues.insert(matrix(i, j));
    }
  }
}

void CHECK_MATRIX_VALUES_IN_RANGE(Eigen::MatrixXd matrix, double min,
                                  double max) {
  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < matrix.rows(); j++) {
      REQUIRE(matrix(i, j) < max);
      REQUIRE(matrix(i, j) > min);
    }
  }
}

// Custom test assertion to check it two matrices are approximately equal
void CHECK_MATRIX_APPROX(const Eigen::MatrixXd &matA,
                         const Eigen::MatrixXd &matB, double epsilon = 1e-6) {
  assert(matA.rows() == matB.rows() && matA.cols() == matB.cols());

  for (int i = 0; i < matA.rows(); ++i) {
    for (int j = 0; j < matA.cols(); ++j) {
      CHECK(std::abs(matA(i, j) - matB(i, j)) < epsilon);
    }
  }
}
