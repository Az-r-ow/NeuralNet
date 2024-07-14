#include <pybind11/pybind11.h>

#include "data/TrainingData.hpp"

namespace py = pybind11;

template <typename Inputs, typename Labels>
void bindTrainingData(py::module &m, const char *name, const char *docstring) {
  py::class_<TrainingData<Inputs, Labels>>(m, name)
      .def(py::init<Inputs, Labels, Inputs, Labels>(), py::arg("inputs"),
           py::arg("labels"), py::arg("testInputs") = Inputs(),
           py::arg("testLabels") = Labels(), docstring)
      .def("batch", &TrainingData<Inputs, Labels>::batch, py::arg("batchSize"),
           py::arg("stratified") = false, py::arg("shuffle") = false,
           py::arg("dropLast") = false, py::arg("verbose") = false,
           "This method will separate the inputs and labels data into batches "
           "of the specified size")
      .def("getMiniBatches", &TrainingData<Inputs, Labels>::getMiniBatches);
};