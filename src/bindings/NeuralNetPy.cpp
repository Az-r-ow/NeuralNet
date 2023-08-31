/**
 * In this file the essential classes and methods are exposed to Python.
 * Just enough to be able to setup and manipulate the Neural Network
 * When the project is built with the PYBIND_BUILD option, it will create a .so file
 * in the build folder.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Network.hpp"
#include "Network.cpp"
#include "Layer.hpp"
#include "Layer.cpp"
#include "utils/Enums.hpp"

namespace py = pybind11;

using namespace NeuralNet;

PYBIND11_MODULE(NeuralNetPy, m)
{
    py::enum_<ActivationName>(m, "ActivationName")
        .value("RELU", ActivationName::RELU)
        .value("SIGMOID", ActivationName::SIGMOID);

    py::enum_<WeightInit>(m, "WeightInit")
        .value("RANDOM", WeightInit::RANDOM)
        .value("GLOROT", WeightInit::GLOROT)
        .value("HE", WeightInit::HE)
        .value("LACUN", WeightInit::LACUN);

    py::class_<Layer>(m, "Layer")
        .def(py::init<int, ActivationName, WeightInit, int>(),
             py::arg("nNeurons"),
             py::arg("activationFunc") = ActivationName::SIGMOID,
             py::arg("weightInit") = WeightInit::RANDOM,
             py::arg("bias") = 0)
        .def("getNumNeurons", &Layer::getNumNeurons);

    py::class_<Network>(m, "Network")
        .def(py::init<double>(),
             py::arg("alpha") = 0.1)
        .def("addLayer", &Network::addLayer)
        .def("getLayer", &Network::getLayer, py::return_value_policy::copy)
        .def("getNumLayers", &Network::getNumLayers)
        .def("train", &Network::train);
}
