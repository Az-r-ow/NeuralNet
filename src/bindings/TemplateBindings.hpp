#include <pybind11/pybind11.h>
#include "data/TrainingData.hpp"

namespace py = pybind11;

template <typename Inputs, typename Labels>
void bindTrainingData(py::module &m, const char *name, const char *docstring)
{
  py::class_<TrainingData<Inputs, Labels>>(m, name)
      .def(py::init<Inputs, Labels>(),
           py::arg("inputs_data"),
           py::arg("labels_data"),
           docstring)
      .def("batch", &TrainingData<Inputs, Labels>::batch, py::arg("batchSize"), "This method will separate the inputs and labels data into batches of the specified size");
};