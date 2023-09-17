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
    py::enum_<ACTIVATION>(m, "ACTIVATION")
        .value("RELU", ACTIVATION::RELU)
        .value("SIGMOID", ACTIVATION::SIGMOID)
        .value("SOFTMAX", ACTIVATION::SOFTMAX);

    py::enum_<WEIGHT_INIT>(m, "WEIGHT_INIT")
        .value("RANDOM", WEIGHT_INIT::RANDOM)
        .value("GLOROT", WEIGHT_INIT::GLOROT)
        .value("HE", WEIGHT_INIT::HE)
        .value("LECUN", WEIGHT_INIT::LECUN);

    py::enum_<LOSS>(m, "LOSS")
        .value("QUADRATIC", LOSS::QUADRATIC)
        .value("MCE", LOSS::MCE);

    py::class_<Layer>(m, "Layer")
        .def(py::init<int, ACTIVATION, WEIGHT_INIT, int>(),
             py::arg("nNeurons"),
             py::arg("activationFunc") = ACTIVATION::SIGMOID,
             py::arg("weightInit") = WEIGHT_INIT::RANDOM,
             py::arg("bias") = 0)
        .def("getNumNeurons", &Layer::getNumNeurons);

    py::class_<Network>(m, "Network")
        .def(py::init<double, int, LOSS>(),
             py::arg("alpha") = 0.1,
             py::arg("epochs") = 10,
             py::arg("loss") = LOSS::QUADRATIC)
        .def("addLayer", &Network::addLayer)
        .def("setBatchSize", &Network::setBatchSize)
        .def("getLayer", &Network::getLayer, py::return_value_policy::copy)
        .def("getNumLayers", &Network::getNumLayers)
        .def("train", &Network::train)
        .def("predict", &Network::predict);
}
