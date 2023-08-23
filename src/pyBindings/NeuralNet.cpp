#include <pybind11/pybind11.h>
#include "../NeuralNet/Network.hpp"

namespace py = pybind11;
using namespace NeuralNet;

PYBIND11_MODULE(NeuralNetPy, m)
{
  py::class_<Network>(m, "Network")
      .def(py::init<double>())
      .def("addLayer", &Network::addLayer)
      .def("getLayer", &Network::getLayer)
      .def("getNumLayers", &Network::getNumLayers)
      .def("train", &Network::train)
      .def("predict", &Network::predict);

  py::class_<Layer> layer(m, "Layer");

  py::enum_<ActivationName>(layer, "ActivationName")
      .value("RELU", ActivationName::RELU)
      .value("SIGMOID", ActivationName::SIGMOID);

  py::enum_<WeightInit>(layer, "WeightInit")
      .value("RANDOM", WeightInit::RANDOM)
      .value("GLOROT", WeightInit::GLOROT)
      .value("HE", WeightInit::HE)
      .value("LACUN", WeightInit::LACUN);

  layer.def(py::init<int, ActivationName, WeightInit, int>(),
            py::arg("nNeurons"),
            py::arg("activationFun") = ActivationName::SIGMOID,
            py::arg("weightInit") = WeightInit::RANDOM,
            py::arg("bias"))
      .def("getNumNeurons", &Layer::getNumNeurons);
}