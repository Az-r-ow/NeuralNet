#include <pybind11/pybind11.h>
#include "data/TrainingData.hpp"

namespace py = pybind11;

template <typename Inputs, typename Labels>
void bindTrainingData(py::module &m, const char *name)
{
  py::class_<TrainingData<Inputs, Labels>>(m, name)
      .def(py::init<Inputs, Labels>())
      .def("batch", &TrainingData<Inputs, Labels>::batch);
};