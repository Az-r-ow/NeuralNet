/**
 * In this file the essential classes and methods are exposed to Python.
 * Just enough to be able to setup and manipulate the Neural Network
 * When the project is built with the PYBIND_BUILD option, it will create a .so
 * file in the build folder.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Model.hpp"
#include "Network.cpp"
#include "Network.hpp"
#include "TemplateBindings.hpp"  // Template classes binding functions
#include "callbacks/CSVLogger.hpp"
#include "callbacks/Callback.hpp"
#include "callbacks/EarlyStopping.hpp"
#include "callbacks/ModelCheckpoint.hpp"
#include "layers/Dense.hpp"
#include "layers/Flatten.hpp"
#include "layers/Layer.hpp"
#include "optimizers/Optimizer.hpp"
#include "optimizers/optimizers.hpp"
#include "utils/Enums.hpp"

namespace py = pybind11;

using namespace NeuralNet;

PYBIND11_MODULE(NeuralNetPy, m) {
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
      .value("RANDOM", WEIGHT_INIT::RANDOM,
             "Initialize weights with random values")
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
      .value("MCE", LOSS::MCE)
      .value("BCE", LOSS::BCE);

  py::module optimizers_m = m.def_submodule("optimizers", R"pbdoc(
      Optimizers
      ----------

      Optimizers are algorithms or methods used to change the attributes of the Neural Network such as weights and learning rate in order to reduce the losses. They are used to solve the optimization problem of minimizing the loss function.

      .. currentmodule:: NeuralNetPy.optimizers
      .. autosummary::
          :toctree: _generate
          :recursive:
    )pbdoc");

  py::class_<Optimizer, std::shared_ptr<Optimizer>>(optimizers_m, "Optimizer");

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(optimizers_m, "SGD", R"pbdoc(
        For more information on `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`

        :param alpha: The learning rate, defaults to 0.001
        :type alpha: float
      )pbdoc")
      .def(py::init<double>(), py::arg("alpha") = 0.001);

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(optimizers_m, "Adam",
                                                     R"pbdoc(
        For more information on `Adam optimizer <https://arxiv.org/abs/1412.6980>`

        :param alpha: The learning rate, defaults to 0.001
        :type alpha: float
        :param beta1: The exponential decay rate for the first moment estimates, defaults to 0.9
        :type beta1: float
        :param beta2: The exponential decay rate for the second-moment estimates, defaults to 0.999
        :type beta2: float
        :param epsilon: A small constant for numerical stability, defaults to 10E-8
        :type epsilon: float
      )pbdoc")
      .def(py::init<double, double, double, double>(), py::arg("alpha") = 0.001,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
           py::arg("epsilon") = 10E-8);

  py::module layers_m = m.def_submodule("layers", R"pbdoc(
      Layers
      ------

      Layers are the building blocks of a Neural Network. They are the individual neurons that are connected to each other to form the network. Each layer has a specific number of neurons and an activation function.

      .. currentmodule:: NeuralNetPy.layers
      .. autosummary::
          :toctree: _generate
          :recursive:
    )pbdoc");

  py::class_<Layer, std::shared_ptr<Layer>>(layers_m, "Layer");

  py::class_<Dense, Layer, std::shared_ptr<Dense>>(layers_m, "Dense", R"pbdoc(
        Initializes a ``Dense`` layer, which is the backbone of a Neural Network.

        :param nNeurons: The number of neurons in the layer
        :type nNeurons: int
        :param activationFunc: The activation function to be used, defaults to ``SIGMOID``
        :type activationFunc: ACTIVATION
        :param weightInit: The weight initialization method to be used, defaults to ``RANDOM``
        :type weightInit: WEIGHT_INIT
        :param bias: The bias to be used, defaults to 0
        :type bias: int

        .. highlight: python 
        .. code-block:: python
            :caption: Example

                import NeuralNetPy as NNP

                layer = NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE)
      )pbdoc")
      .def(py::init<int, ACTIVATION, WEIGHT_INIT, int>(), py::arg("nNeurons"),
           py::arg("activationFunc") = ACTIVATION::SIGMOID,
           py::arg("weightInit") = WEIGHT_INIT::RANDOM, py::arg("bias") = 0)
      .def("typeStr", &Flatten::typeStr, R"pbdoc(
        Returns the type of the layer.
      )pbdoc");

  py::class_<Dropout, Layer, std::shared_ptr<Dropout>>(layers_m, "Dropout",
                                                       R"pbdoc(
    Initializes a ``Dropout`` layer, it's a layer that simply applies a dropout to the input.

    :param rate: A float between 0 and 1. It represents the fraction of the inputs to drop.
    :type rate: float32
    :param seed: An integer used as random seed. If not provided a random seed will be generated.
    :type seed: int
  )pbdoc")
      .def(py::init<float, int>(), py::arg("rate"), py::arg("seed") = 0)
      .def("typeStr", &Flatten::typeStr, R"pbdoc(
        Returns the type of the layer.
      )pbdoc");

  py::class_<Flatten, Layer, std::shared_ptr<Flatten>>(layers_m, "Flatten",
                                                       R"pbdoc(
        Initializes a ``Flatten`` layer. The sole purpose of this layer is to vectorize matrix inputs like images.

        :param inputShape: The shape of the input matrix (rows, cols or number of pixels per row and column in the case of images)
        :type inputShape: tuple

        .. code-block:: python
            :caption: Example

                import NeuralNetPy as NNP

                layer = NNP.layers.Flatten((3, 3))
      )pbdoc")
      .def(py::init<std::tuple<int, int>>(), py::arg("inputShape"))
      .def("typeStr", &Flatten::typeStr, R"pbdoc(
        Returns the type of the layer.
      )pbdoc");

  py::bind_vector<std::vector<std::shared_ptr<Flatten>>>(layers_m,
                                                         "VectorFlatten");
  py::bind_vector<std::vector<std::shared_ptr<Dense>>>(layers_m, "VectorDense");
  py::bind_vector<std::vector<std::shared_ptr<Dropout>>>(layers_m,
                                                         "VectorDropout");

  py::module callbacks_m = m.def_submodule("callbacks", R"pbdoc(
      Callbacks
      ----------

      Callbacks are a set of functions that can be applied at given stages of the training procedure. They can be used to get a view on internal states and statistics of the model during training. You can pass a list of callbacks to the ``train`` method of the ``Network`` class.
      Each callback has it's own purpose make sure the read their documentation carefully. 

      .. currentmodule:: NeuralNetPy.callbacks
      .. autosummary::
          :toctree: _generate
          :recursive:
    )pbdoc");

  py::class_<Callback, std::shared_ptr<Callback>>(callbacks_m, "Callback",
                                                  R"pbdoc(
      This is the base class for all callbacks.
    )pbdoc");

  py::class_<EarlyStopping, Callback, std::shared_ptr<EarlyStopping>>(
      callbacks_m, "EarlyStopping", R"pbdoc(
          Initializes an ``EarlyStopping`` callback. This callback will stop the training if the given metric doesn't improve more than the given delta over a certain number of epochs (patience).

          :param metric: The metric to be monitored (Either ``LOSS`` or ``ACCURACY``), defaults to ``LOSS``
          :type metric: str
          :param minDelta: The minimum change in the monitored metric to be considered an improvement, defaults to 0.01
          :type minDelta: float
          :param patience: The number of epochs with no improvement after which training will be stopped, defaults to 0
          :type patience: int

          .. highlight: python
          .. code-block:: python
              :caption: Example

              network.train(X, y, 100, [NNP.callbacks.EarlyStopping("loss", 0.01, 10)])
      )pbdoc")
      .def(py::init<std::string, double, int>(), py::arg("metric") = "LOSS",
           py::arg("minDelta") = 0.01, py::arg("patience") = 0);

  py::class_<CSVLogger, Callback, std::shared_ptr<CSVLogger>>(
      callbacks_m, "CSVLogger", R"pbdoc(
        Initializes a ``CSVLogger`` callback. This callback will log the training process in a CSV file.

        .. highlight: python
        .. code-block:: python
            :caption: Example

            network.train(inputs, labels, 100, [NNP.callbacks.CSVLogger("logs.csv")])
      )pbdoc")
      .def(py::init<std::string, std::string>(), py::arg("filename"),
           py::arg("separator") = ",");

  py::class_<ModelCheckpoint, Callback, std::shared_ptr<ModelCheckpoint>>(
      callbacks_m, "ModelCheckpoint", R"pbdoc(
      ``ModelCheckpoint`` callback is used in parallel of training `model.train` to save a model (it's parameters) in a checkpoint file at a given interval.

      A couple of options provided by the callbacks are : 
      * ``saveBestOnly`` If activated will save the "best" model (which is deduced automatically).
      * ``numEpochs`` Number of epoch intervals between checkpoints (only valid if ``saveBestOnly`` is ``False``)
      
      Params
      ======

      :param folderPath: The path to the folder in which to save the checkpoints.
      :type folderPath: str
      :param saveBestOnly: Whether to save the best checkpoint or each one of them (default: ``True``)
      :type saveBestOnly: bool
      :param numEpochs: The number of epochs interval between checkpoints
      :type numEpochs: int
      :param verbose: Verbose output (default: False)
      :type verbose: bool
    )pbdoc")
      .def(py::init<std::string, bool, int, bool>(), py::arg("folderPath"),
           py::arg("saveBestOnly") = true, py::arg("numEpochs") = 1,
           py::arg("verbose") = false);

  py::bind_vector<std::vector<std::shared_ptr<Callback>>>(callbacks_m,
                                                          "VectorCallback");
  py::bind_vector<std::vector<std::shared_ptr<EarlyStopping>>>(
      callbacks_m, "VectorEarlyStopping");
  py::bind_vector<std::vector<std::shared_ptr<CSVLogger>>>(callbacks_m,
                                                           "VectorCSVLogger");
  py::bind_vector<std::vector<std::shared_ptr<ModelCheckpoint>>>(
      callbacks_m, "VectorModelCheckpoint");

  // TrainingData with 2 dimensional inputs
  bindTrainingData<std::vector<std::vector<double>>, std::vector<double>>(
      m, "TrainingData2dI", R"pbdoc(
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
  bindTrainingData<std::vector<std::vector<std::vector<double>>>,
                   std::vector<double>>(m, "TrainingData3dI", R"pbdoc(
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
   * This is why I had to specify the type "Network", I'll have to do so for
   * every type added
   */

  py::module models_m = m.def_submodule("models", R"pbdoc(
      Models
      ------

      Models are used in machine learning to make predictions or decisions without being explicitly programmed to do so. 

      .. currentmodule:: NeuralNetPy.models
      .. autosummary::
          :toctree: _generate
          :recursive:
    )pbdoc");

  py::class_<Model>(models_m, "Model", "Base class for all models")
      .def_static("save_to_file", &Model::save_to_file<Network>, R"pbdoc(
        This function will save the given ``Model``'s parameters in a binary file.

        .. highlight: python
        .. code-block:: python
            :caption: Usage

            import NeuralNetPy as NNP

            network = NNP.models.Network()
            network.setup(optimizer=NNP.optimizers.SGD(0.01))
            network.addLayer(NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.layers.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

            # ... after training

            NNP.models.Model.save_to_file(network, "network.bin") 

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
            network = NNP.models.Network()

            # Populate it with previously saved parameters
            NNP.models.Model.load_from_file("network.bin", network)
      )pbdoc");

  py::class_<Network, Model>(models_m, "Network", R"pbdoc(
      This is the base of a Neural Network. You can setup the network with the given optimizer and loss function.

      :param optimizer: The optimizer to be used from the ``optimizers`` module
      :type optimizer: Optimizer
      :param loss: The loss function to be used from the ``LOSS`` enum, defaults to ``QUADRATIC``
      :type loss: LOSS

      .. highlight: python
      .. code-block:: python
          :caption: Example

          import NeuralNetPy as NNP

          network = NNP.models.Network()
          network.setup(optimizer=NNP.SGD(0.01), loss=NNP.LOSS.MCQ)
      )pbdoc")
      .def(py::init<>())
      .def("getSlug", &Network::getSlug)
      .def("setup", &Network::setup, py::arg("optimizer"),
           py::arg("loss") = LOSS::QUADRATIC)
      .def("addLayer", &Network::addLayer, R"pbdoc(
            Add a layer to the network. 

            :param layer: The layer to be added
            :type layer: Layer

            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.models.Network()
                network.addLayer(NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))

            .. warning::
                The order of the layers added is important, it will reflect the overall structure of the network.
            
            .. danger::
                Under no circumstances you should add a ``Flatten`` layer as a hidden layer.
           )pbdoc")
      .def("getLayer", &Network::getLayer, py::return_value_policy::copy,
           R"pbdoc(
            Get a layer from the network by it's index. They're 0-indexed.

            :param index: The index of the layer
            :type index: int
            :return: The layer at the given index
            :rtype: Layer

            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.models.Network()
                network.addLayer(NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
                network.addLayer(NNP.layers.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

                layer = network.getLayer(1) # Return Dense layer with 2 neurons
          )pbdoc")
      .def("getNumLayers", &Network::getNumLayers,
           "Return the number of layers in the network.")
      .def("train",
           static_cast<double (Network::*)(
               std::vector<std::vector<double>>, std::vector<double>, int,
               const std::vector<std::shared_ptr<Callback>>, bool)>(
               &Network::train),
           py::arg("inputs"), py::arg("targets"), py::arg("epochs"),
           py::arg("callbacks") = std::vector<std::shared_ptr<Callback>>(),
           py::arg("progBar") = true,
           R"pbdoc(
            Train the network by passing it 2 dimensional inputs (vectors).

            :param inputs: A list of vectors representing the inputs
            :type inputs: list[list[float]]
            :param labels: A list of labels
            :type labels: list[float]
            :param epochs: The number of epochs to train the network
            :type epochs: int
            :param callbacks: A list of callbacks to be used during the training
            :type callbacks: list[Callback]
            :param progBar: Whether or not to enable the progress bar
            :type progBar: bool
            :return: The average loss throughout the training
            :rtype: float

            .. highlight: python
            .. code-block:: python
                :caption: Example

                import NeuralNetPy as NNP

                network = NNP.models.Network()
                network.setup(optimizer=NNP.optimizers.SGD(0.01), loss=NNP.LOSS.MCQ)
                network.addLayer(NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
                network.addLayer(NNP.layers.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

                inputs = [
                  [0.4, 0.5, 0.67],
                  [0.3, 0.2, 0.1],
                  [0.1, 0.2, 0.3]
                ]

                labels = [1, 0, 1]

                loss = network.train(inputs, labels, 10)
            
        )pbdoc")
      .def("train",
           static_cast<double (Network::*)(
               std::vector<std::vector<std::vector<double>>>,
               std::vector<double>, int,
               const std::vector<std::shared_ptr<Callback>>, bool)>(
               &Network::train),
           py::arg("inputs"), py::arg("targets"), py::arg("epochs"),
           py::arg("callbacks") = std::vector<std::shared_ptr<Callback>>(),
           py::arg("progBar") = true,
           R"pbdoc(
        Train the network by passing it a list of 3 dimensional inputs (matrices).

        :param inputs: A list of matrices representing the inputs
        :type inputs: list[list[list[float]]]
        :param labels: A list of labels
        :type labels: list[float]
        :param epochs: The number of epochs to train the network
        :type epochs: int
        :param callbacks: A list of callbacks to be used during the training
        :type callbacks: list[Callback]
        :param progBar: Whether or not to enable the progress bar
        :type progBar: bool
        :return: The average loss throughout the training
        :rtype: float

        .. highlight: python
        .. code-block: python
            :caption: Example

            import NeuralNetPy as NNP

            network = NNP.models.Network()
            network.setup(optimizer=NNP.optimizers.SGD(0.01), loss=NNP.LOSS.MCQ)
            network.addLayer(NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.layers.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

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
      .def("train",
           static_cast<double (Network::*)(
               TrainingData<std::vector<std::vector<double>>,
                            std::vector<double>>,
               int, const std::vector<std::shared_ptr<Callback>>, bool)>(
               &Network::train),
           py::arg("trainingData"), py::arg("epochs"),
           py::arg("callbacks") = std::vector<std::shared_ptr<Callback>>(),
           py::arg("progBar") = true,
           R"pbdoc(
        Train the network by passing it a ``TrainingData2dI`` object.

        :param trainingData: A ``TrainingData2dI`` object
        :type trainingData: TrainingData2dI
        :param epochs: The number of epochs to train the network
        :type epochs: int
        :param callbacks: A list of callbacks to be used during the training
        :type callbacks: list[Callback]
        :param progBar: Whether or not to enable the progress bar
        :type progBar: bool
        :return: The average loss throughout the training
        :rtype: float

        .. highlight: python
        .. code-block: python
            :caption: Example

            import NeuralNetPy as NNP

            network = NNP.models.Network()
            network.setup(optimizer=NNP.optimizers.SGD(0.01), loss=NNP.LOSS.MCQ)
            network.addLayer(NNP.layers.Dense(3, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
            network.addLayer(NNP.layers.Dense(2, NNP.ACTIVATION.SIGMOID, NNP.WEIGHT_INIT.HE))

            # Meaningless values just for the sake of the example
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
      .def("train",
           static_cast<double (Network::*)(
               TrainingData<std::vector<std::vector<std::vector<double>>>,
                            std::vector<double>>,
               int, const std::vector<std::shared_ptr<Callback>>, bool)>(
               &Network::train),
           py::arg("trainingData"), py::arg("epochs"),
           py::arg("callbacks") = std::vector<std::shared_ptr<Callback>>(),
           py::arg("progBar") = true,
           R"pbdoc(
        Train the network by passing it a ``TrainingData3dI`` object.

        :param trainingData: A ``TrainingData3dI`` object
        :type trainingData: TrainingData3dI
        :param epochs: The number of epochs to train the network
        :type epochs: int
        :param callbacks: A list of callbacks to be used during the training
        :type callbacks: list[Callback]
        :param progBar: Whether or not to enable the progress bar
        :type progBar: bool
        :return: The average loss throughout the training
        :rtype: float

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
      .def("predict",
           static_cast<Eigen::MatrixXd (Network::*)(
               std::vector<std::vector<double>>)>(&Network::predict),
           R"pbdoc(
        Feed forward the given inputs through the network and return the predictions/outputs.

        :param inputs: A list of vectors representing the inputs
        :type inputs: list[list[float]]
        :return: A matrix representing the outputs of the network for the given inputs
        :rtype: numpy.ndarray
      )pbdoc")
      .def("predict",
           static_cast<Eigen::MatrixXd (Network::*)(
               std::vector<std::vector<std::vector<double>>>)>(
               &Network::predict),
           R"pbdoc(
        Feed forward the given inputs through the network and return the predictions/outputs.

        :param inputs: A list of vectors representing the inputs
        :type inputs: list[list[list[float]]]
        :return: A matrix representing the outputs of the network for the given inputs
        :rtype: numpy.ndarray
      )pbdoc");
}
