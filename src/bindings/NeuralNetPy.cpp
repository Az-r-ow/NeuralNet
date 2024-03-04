/**
 * In this file the essential classes and methods are exposed to Python.
 * Just enough to be able to setup and manipulate the Neural Network
 * When the project is built with the PYBIND_BUILD option, it will create a .so file
 * in the build folder.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "Network.hpp"
#include "Model.hpp"
#include "Network.cpp"
#include "layers/Layer.hpp"
#include "layers/Flatten.hpp"
#include "layers/Dense.hpp"
#include "optimizers/Optimizer.hpp"
#include "optimizers/optimizers.hpp"
#include "callbacks/Callback.hpp"
#include "callbacks/EarlyStopping.hpp"
#include "callbacks/CSVLogger.hpp"
#include "utils/Enums.hpp"
#include "TemplateBindings.hpp" // Template classes binding functions

namespace py = pybind11;

using namespace NeuralNet;

PYBIND11_MODULE(NeuralNetPy, m)
{
  m.doc() = R"pbdoc(
      Neural Network Library
      -----------------------
      .. currentmodule:: NeuralNetPy
      .. autosummary::
          :toctree: _generate
    )pbdoc";

  py::enum_<ACTIVATION>(m, "ACTIVATION")
      .value("RELU", ACTIVATION::RELU, "Rectified Activation Function")
      .value("SIGMOID", ACTIVATION::SIGMOID, "Sigmoid Activation Function")
      .value("SOFTMAX", ACTIVATION::SOFTMAX, "Softmax Activation Function");

  py::enum_<WEIGHT_INIT>(m, "WEIGHT_INIT")
      .value("RANDOM", WEIGHT_INIT::RANDOM, "Initialize weights with random values")
      .value("GLOROT", WEIGHT_INIT::GLOROT, R"pbdoc(
          Initialize weights with Glorot initialization.

          .. tip ::
              Best when combined with RELU
          )pbdoc")
      .value("HE", WEIGHT_INIT::HE, R"pbdoc(
          Initialize weights with He initialization.

          .. tip::
              Best when combined with RELU or SOFTMAX
          )pbdoc")
      .value("LECUN", WEIGHT_INIT::LECUN, R"pbdoc(
          Initialize weights with Lecun initialization.

          .. tip::
              Best when combined with SOFTMAX
          )pbdoc");

  py::enum_<LOSS>(m, "LOSS")
      .value("QUADRATIC", LOSS::QUADRATIC)
      .value("MCE", LOSS::MCE);

  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
      .def(py::init<double>(),
           py::arg("alpha") = 0.001,
           R"pbdoc(
                For more information on `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`

                .. highlight: python
                .. code-block:: python
                    :caption: Example

                        import NeuralNetPy as NNP

                        network = NNP.Network()
                        network.setup(optimizer=NNP.SGD(0.01))
              )pbdoc");

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam")
      .def(py::init<double, double, double, double>(),
           py::arg("alpha") = 0.001,
           py::arg("beta1") = 0.9,
           py::arg("beta2") = 0.999,
           py::arg("epsilon") = 10E-8,
           R"pbdoc(
                For more information on `Adam optimizer <https://arxiv.org/abs/1412.6980>`
                
                .. highlight: python 
                .. code-block:: python
                    :caption: Example

                        import NeuralNetPy as NNP

                        network = NNP.Network()
                        network.setup(optimizer=NNP.Adam(0.001))
             )pbdoc");

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def(py::init<int, ACTIVATION, WEIGHT_INIT, int>(),
           py::arg("nNeurons"),
           py::arg("activationFunc") = ACTIVATION::SIGMOID,
           py::arg("weightInit") = WEIGHT_INIT::RANDOM,
           py::arg("bias") = 0,
           R"pbdoc(
                Base class of all layers.

                .. tip::
                    It is recommended that you use it's derivatives instead. Like `Dense` or `Flatten` since they're more specific.
             )pbdoc")
      .def("getNumNeurons", &Layer::getNumNeurons);

  py::class_<Dense, Layer, std::shared_ptr<Dense>>(m, "Dense")
      .def(py::init<int, ACTIVATION, WEIGHT_INIT, int>(),
           py::arg("nNeurons"),
           py::arg("activationFunc") = ACTIVATION::SIGMOID,
           py::arg("weightInit") = WEIGHT_INIT::RANDOM,
           py::arg("bias") = 0,
           R"pbdoc(
                Initializes a ``Dense`` layer, which is the backbone of a Neural Network.

                .. highlight: python 
                .. code-block:: python
                    :caption: Example

                        import NeuralNetPy as NNP

                        layer = NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE)
             )pbdoc");

  py::class_<Flatten, Layer, std::shared_ptr<Flatten>>(m, "Flatten")
      .def(py::init<std::tuple<int, int>>(),
           py::arg("inputShape"),
           R"pbdoc(
                Initializes a ``Flatten`` layer. The sole purpose of this layer is to vectorize matrix inputs like images.

                .. code-block:: python
                    :caption: Example

                        import NeuralNetPy as NNP

                        layer = NNP.Flatten((3, 3))
             )pbdoc");

  py::bind_vector<std::vector<std::shared_ptr<Layer>>>(m, "VectorLayer");
  py::bind_vector<std::vector<std::shared_ptr<Flatten>>>(m, "VectorFlatten");
  py::bind_vector<std::vector<std::shared_ptr<Dense>>>(m, "VectorDense");

  py::class_<Callback, std::shared_ptr<Callback>>(m, "Callback");

  py::class_<EarlyStopping, Callback, std::shared_ptr<EarlyStopping>>(m, "EarlyStopping")
      .def(py::init<std::string, double, int>(),
           py::arg("metric"),
           py::arg("minDelta") = 0.01,
           py::arg("patience") = 0,
           R"pbdoc(
                Initializes an ``EarlyStopping`` callback. This callback will stop the training if the given metric doesn't improve more than the given delta over a certain number of epochs (patience).

                .. highlight: python
                .. code-block:: python
                    :caption: Example

                    import NeuralNetPy as NNP

                    network = NNP.Network()
                    network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
                    network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
                    network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

                    inputs = [
                      [0.4, 0.5, 0.67],
                      [0.3, 0.2, 0.1],
                      [0.1, 0.2, 0.3]
                    ]

                    labels = [1, 0, 1]

                    earlyStopping = NNP.EarlyStopping("loss", 0.01, 10)

                    network.train(inputs, labels, 100, [earlyStopping])
             )pbdoc");

  py::class_<CSVLogger, Callback, std::shared_ptr<CSVLogger>>(m, "CSVLogger")
      .def(py::init<std::string, std::string>(),
           py::arg("filename"),
           py::arg("separator") = ",",
           R"pbdoc(
                Initializes a ``CSVLogger`` callback. This callback will log the training process in a CSV file.

                .. highlight: python
                .. code-block:: python
                    :caption: Example

                    import NeuralNetPy as NNP

                    network = NNP.Network()
                    network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
                    network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
                    network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

                    inputs = [
                      [0.4, 0.5, 0.67],
                      [0.3, 0.2, 0.1],
                      [0.1, 0.2, 0.3]
                    ]

                    labels = [1, 0, 1]

                    csvLogger = NNP.CSVLogger("logs.csv")

                    network.train(inputs, labels, 100, [csvLogger])
             )pbdoc");

  py::bind_vector<std::vector<std::shared_ptr<Callback>>>(m, "VectorCallback");
  py::bind_vector<std::vector<std::shared_ptr<EarlyStopping>>>(m, "VectorEarlyStopping");

  // TrainingData with 2 dimensional inputs
  bindTrainingData<std::vector<std::vector<double>>, std::vector<double>>(m, "TrainingData2dI", R"pbdoc(
    Represents training data with 2 dimensional inputs (vectors). This class is supposed to bring the table some methods to easily manipulate the data and prepare it for training.

    .. highlight: python 
    .. code-block:: python
        :caption: Example

        import numpy as np
        import NeuralNetPy as NNP

        inputs = np.array([
          [0.4, 0.5, 0.67],
          [0.3, 0.2, 0.1],
          [0.1, 0.2, 0.3]
        ])

        # Labels are the same lengths as the inputs
        labels = np.array([1, 0, 1])

        trainingData = NNP.TrainingData2dI(inputs, labels)

    .. warning::
        The inputs and labels must be of the same length
  )pbdoc");

  // TrainingData with 3 dimensional inputs
  bindTrainingData<std::vector<std::vector<std::vector<double>>>, std::vector<double>>(m, "TrainingData3dI", R"pbdoc(
    Represents training data with 3 dimensional inputs (matrices). This class is supposed to bring the table some methods to easily manipulate the data and prepare it for training.

    .. highlight: python
    .. code-block:: python
        :caption: Example
          
          import numpy as np
          import NeuralNetPy as NNP
  
          inputs = np.array([
            [
              [0.4, 0.5, 0.67],
              [0.3, 0.2, 0.1],
              [0.1, 0.2, 0.3]
            ],
            [
              [0.4, 0.5, 0.67],
              [0.3, 0.2, 0.1],
              [0.1, 0.2, 0.3]
            ]
          ])
  
          # Labels are the same lengths as the inputs
          labels = np.array([1, 0])
  
          trainingData = NNP.TrainingData3dI(inputs, labels)

    .. warning::
        The inputs and labels must be of the same length.
  )pbdoc");

  /**
   * > You can only bind explicitly instantiated versions of your function
   *
   * https://github.com/pybind/pybind11/issues/199#issuecomment-220302516
   *
   * This is why I had to specify the type "Network", I'll have to do so for every type added
   */
  py::class_<Model>(m, "Model")
      .def_static("save_to_file", &Model::save_to_file<Network>, R"pbdoc(
        This function will save the given ``Model``'s parameters in a binary file.

        .. highlight: python
        .. code-block:: python
            :caption: Usage

            import NeuralNetPy as NNP

            network = NNP.Network()
            network.setup(optimizer=NNP.SGD(0.01))
            network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

            # ... after training

            NNP.Model.save_to_file(network, "network.bin") 

        .. warning::
            The file content will be overwritten if it already exists.
      )pbdoc")
      .def_static("load_from_file", &Model::load_from_file<Network>, R"pbdoc(
        This function will load the parameters of a ``Model`` from a binary file.

        .. highlight: python
        .. code-block:: python
            :caption: Usage

            import NeuralNetPy as NNP

            # Initialize an empty network            
            network = NNP.Network()

            # Populate it with previously saved parameters
            NNP.Model.load_from_file("network.bin", network)
      )pbdoc");

  py::class_<Network, Model>(m, "Network")
      .def(py::init<>(), R"pbdoc(
            Initializes a Neural Network
        )pbdoc")
      .def("setup",
           &Network::setup,
           py::arg("optimizer"),
           py::arg("loss") = LOSS::QUADRATIC,
           R"pbdoc(
            Setup the network with the given optimizer and loss function

            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.Network()
                network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
           )pbdoc")
      .def("addLayer", &Network::addLayer, R"pbdoc(
            Add a layer to the network. 


            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.Network()
                network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))

            .. warning::
                The order of the layers added is important, it will reflect the overall structure of the network.
            
            .. danger::
                Under no circumstances you should add a ``Flatten`` layer as a hidden layer.
           )pbdoc")
      .def("getLayer",
           &Network::getLayer,
           py::return_value_policy::copy,
           R"pbdoc(
            Get a layer from the network by it's index. They're 0-indexed.

            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.Network()
                network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
                network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

                layer = network.getLayer(1) # Return Dense layer with 2 neurons
          )pbdoc")
      .def("getNumLayers", &Network::getNumLayers, "Return the number of layers in the network.")
      .def("train", static_cast<double (Network::*)(std::vector<std::vector<double>>, std::vector<double>, int, const std::vector<std::shared_ptr<Callback>>)>(&Network::train), R"pbdoc(
            Train the network by passing it 2 dimensional inputs (vectors).

            :param inputs: A list of vectors representing the inputs
            :param labels: A list of labels
            :param epochs: The number of epochs to train the network
            :return: The average loss throughout the training
            :rtype: float

            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.Network()
                network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
                network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
                network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

                inputs = [
                  [0.4, 0.5, 0.67],
                  [0.3, 0.2, 0.1],
                  [0.1, 0.2, 0.3]
                ]

                labels = [1, 0, 1]

                loss = network.train(inputs, labels, 10)
            
        )pbdoc")
      .def("train", static_cast<double (Network::*)(std::vector<std::vector<std::vector<double>>>, std::vector<double>, int, const std::vector<std::shared_ptr<Callback>>)>(&Network::train), R"pbdoc(
        Train the network by passing it a list of 3 dimensional inputs (matrices).

        :param inputs: A list of matrices representing the inputs
        :param labels: A list of labels
        :param epochs: The number of epochs to train the network
        :return: The average loss throughout the training

        .. highlight: python
        .. code-block: python
            :caption: Example

            import NeuralNetPy as NNP

            network = NNP.Network()
            network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
            network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

            inputs = [
              [
                [0.4, 0.5, 0.67],
                [0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3]
              ],
              [
                [0.4, 0.5, 0.67],
                [0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3]
              ]
            ]

            labels = [1, 0]

            loss = network.train(inputs, labels, 10)
      )pbdoc")
      .def("train", static_cast<double (Network::*)(TrainingData<std::vector<std::vector<double>>, std::vector<double>>, int, const std::vector<std::shared_ptr<Callback>>)>(&Network::train), R"pbdoc(
        Train the network by passing it a ``TrainingData2dI`` object.

        :param trainingData: A ``TrainingData2dI`` object
        :param epochs: The number of epochs to train the network
        :return: The average loss throughout the training

        .. highlight: python
        .. code-block: python
            :caption: Example

            import NeuralNetPy as NNP

            network = NNP.Network()
            network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
            network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

            inputs = [
              [0.4, 0.5, 0.67],
              [0.3, 0.2, 0.1],
              [0.1, 0.2, 0.3]
            ]

            labels = [1, 0, 1]

            trainingData = NNP.TrainingData2dI(inputs, labels)

            # For mini-batch training
            trainingData.batch(2)

            loss = network.train(trainingData, 10)
      )pbdoc")
      .def("train", static_cast<double (Network::*)(TrainingData<std::vector<std::vector<std::vector<double>>>, std::vector<double>>, int, const std::vector<std::shared_ptr<Callback>>)>(&Network::train), R"pbdoc(
        Train the network by passing it a ``TrainingData3dI`` object.

        :param trainingData: A ``TrainingData3dI`` object
        :param epochs: The number of epochs to train the network
        :return: The average loss throughout the training

        .. highlight: python
        .. code-block: python
            :caption: Example

            import NeuralNetPy as NNP

            network = NNP.Network()
            network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
            network.addLayer(NNP.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

            # Meaningless values just for the sake of the example
            inputs = [
              [
                [0.4, 0.5, 0.67],
                [0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3]
              ],
              [
                [0.4, 0.5, 0.67],
                [0.3, 0.2, 0.1],
                [0.1, 0.4, 0.3]
              ],
              [
                [0.4, 0.5, 0.67],
                [0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3]
              ]
            ]

            labels = [1, 0]

            trainingData = NNP.TrainingData3dI(inputs, labels)

            # For mini-batch training
            trainingData.batch(2)

            loss = network.train(trainingData, 10)
      )pbdoc")
      .def("predict", static_cast<Eigen::MatrixXd (Network::*)(std::vector<std::vector<double>>)>(&Network::predict), R"pbdoc(
        Feed forward the given inputs through the network and return the predictions/outputs.

        :param inputs: A list of vectors representing the inputs
        :return: A matrix representing the outputs of the network for the given inputs
        :rtype: numpy.ndarray
      )pbdoc")
      .def("predict", static_cast<Eigen::MatrixXd (Network::*)(std::vector<std::vector<std::vector<double>>>)>(&Network::predict), R"pbdoc(
        Feed forward the given inputs through the network and return the predictions/outputs.

        :param inputs: A list of vectors representing the inputs
        :return: A matrix representing the outputs of the network for the given inputs
        :rtype: numpy.ndarray
      )pbdoc");
}
