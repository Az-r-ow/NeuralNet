#pragma once

#include <Eigen/Dense>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/vector.hpp>
#include "Enums.hpp"

namespace cereal
{
  template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
  void save(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const &m)
  {
    int32_t rows = m.rows();
    int32_t cols = m.cols();
    ar(rows);
    ar(cols);
    ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
  }

  template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
  void load(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &m)
  {
    int32_t rows;
    int32_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);

    ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
  }

  template <class Archive, class T1, class T2>
  void save(Archive &ar, const std::tuple<T1, T2> &t)
  {
    ar(std::get<0>(t), std::get<1>(t));
  }

  template <class Archive, class T1, class T2>
  void load(Archive &ar, std::tuple<T1, T2> &t)
  {
    ar(std::get<0>(t), std::get<1>(t));
  }
} // namespace cereal