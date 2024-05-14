Quick Start
==========

Requirements :
- `Docker`

Start by building the python image with `docker compose` : 

.. code-block:: bash

  docker-compose up 

.. note::
  We're assuming that you're running the commands from the root of the project. 

Access the docker service which will act as your working environment :

.. code-block:: bash

  docker-compose run python-example

To know if you're in you should see a change in the cli's prompt :

.. code-block::
  
  root@ad245b0ff5c4:/app#

Now all you have to do is navigate to the python example project of your liking and run the scripts just like you would normally.
The following example is for running the MNIST example which consists of downloading the handwritten digits dataset. Then, creating a Neural Network model and training it on the dataset.

.. code-block:: bash

  cd examples/train-predict-MNIST && mkdir dataset && python main.py

If you look at the `main.py` file, you'll notice : 

- Fetching the MNIST handwritten digits dataset 
  .. code-block:: python

    if not file_exists(MNIST_DATASET_FILE):
      print("Mnist dataset not found")
      get_MNIST_dataset(MNIST_DATASET_FILE)

- Initiating and composing a Neural Network 

.. code-block:: python 

  network = NNP.models.Network()

  network.addLayer(NNP.layers.Flatten((28, 28)))
  network.addLayer(NNP.layers.Dense(128, NNP.ACTIVATION.RELU, NNP.WEIGHT_INIT.HE))
  network.addLayer(NNP.layers.Dense(10, NNP.ACTIVATION.SOFTMAX, NNP.WEIGHT_INIT.LECUN))

  # Setting up the networks parameters
  network.setup(optimizer=NNP.optimizers.Adam(0.01), loss=NNP.LOSS.MCE)

- Formatting and normalizing the data and then storing it in a `TrainingData3dI` object. Which simplifies batching. 
  
.. code-block:: python

  # preparing the training data
  f_x_train = [normalize_img(x) for x in x_train]

  trainingData = NNP.TrainingData3dI(f_x_train[:NUM_TRAININGS], y_train[:NUM_TRAININGS])

  trainingData.batch(128) # Creating batches of 128 inputs

- Training the data
- Computing the accuracy 
- Serializing the trained model and saving it in a binary format

.. code-block:: python

  # Saving the trained model in a bin file
  NNP.models.Model.save_to_file('./model.bin', network)

  saved_model = NNP.models.Network()

- Loading the model into a new instance of `Network` from the file again (simply for showcase)
  
.. code-block:: python

  # Saving the trained model in a bin file
  NNP.models.Model.save_to_file('./model.bin', network)

  saved_model = NNP.models.Network()

  NNP.models.Model.load_from_file('./model.bin', saved_model)

- Testing the model with the `test_data`

Installation
------------

.. warning::
  The project must be built with ``cmake`` in order to use the python library.

Then inside your python script, you must include the module path to the system path :

.. code-block:: python

    import sys
    sys.path.append('/path/to/build/folder')

Then you can import the module just like any other :

.. code-block:: python

  import NeuralNetPy as NNP

.. Attention::
  The path to the build folder must be set before importing the module.