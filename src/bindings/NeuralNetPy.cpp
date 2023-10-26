/**
 * In this file the essential classes and methods are exposed to Python.
 * Just enough to be able to setup and manipulate the Neural Network
 * When the project is built with the PYBIND_BUILD option, it will create a .so file
 * in the build folder.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "Network.hpp"
#include "Model.hpp"
#include "Network.cpp"
#include "Layer.hpp"
#include "Layer.cpp"
#include "Flatten.hpp"
#include "Flatten.cpp"
#include "interfaces/Optimizer.hpp"
#include "optimizers/optimizers.hpp"
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

    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");

    py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
        .def(py::init<double>());

    py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam")
        .def(py::init<double, double, double, double>(),
             py::arg("alpha") = 0.001,
             py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999,
             py::arg("epsilon") = 10E-8);

    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
        .def(py::init<int, ACTIVATION, WEIGHT_INIT, int>(),
             py::arg("nNeurons"),
             py::arg("activationFunc") = ACTIVATION::SIGMOID,
             py::arg("weightInit") = WEIGHT_INIT::RANDOM,
             py::arg("bias") = 0)
        .def("getNumNeurons", &Layer::getNumNeurons);

    py::class_<Flatten, Layer, std::shared_ptr<Flatten>>(m, "Flatten")
        .def(py::init<std::tuple<int, int>, ACTIVATION, WEIGHT_INIT, int>(),
             py::arg("inputShape"),
             py::arg("activation") = ACTIVATION::SIGMOID,
             py::arg("weightInit") = WEIGHT_INIT::RANDOM,
             py::arg("bias") = 0);

    py::bind_vector<std::vector<std::shared_ptr<Layer>>>(m, "VectorLayer");
    py::bind_vector<std::vector<std::shared_ptr<Flatten>>>(m, "VectorFlatten");

    /**
     * > You can only bind explicitly instantiated versions of your function
     *
     * https://github.com/pybind/pybind11/issues/199#issuecomment-220302516
     *
     * This is why I had to specify the type "Network", I'll have to do so for every type added
     */
    py::class_<Model>(m, "Model")
        .def_static("save_to_file", &Model::save_to_file<Network>)
        .def_static("load_from_file", &Model::load_from_file<Network>);

    py::class_<Network, Model>(m, "Network")
        .def(py::init<double>(),
             py::arg("alpha") = 0.1)
        .def("setup", &Network::setup,
             py::arg("optimizer"),
             py::arg("epochs") = 10,
             py::arg("loss") = LOSS::QUADRATIC)
        .def("addLayer", &Network::addLayer)
        .def("setBatchSize", &Network::setBatchSize)
        .def("getLayer", &Network::getLayer, py::return_value_policy::copy)
        .def("getNumLayers", &Network::getNumLayers)
        .def("train", static_cast<double (Network::*)(std::vector<std::vector<double>>, std::vector<double>)>(&Network::train), "Train the network")
        .def("train", static_cast<double (Network::*)(std::vector<std::vector<std::vector<double>>>, std::vector<double>)>(&Network::train), "Train the network")
        .def("predict", &Network::predict)
        .def("getLayers", &Network::getLayers);
}
