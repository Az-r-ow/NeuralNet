Usage
=====

Quick Start
-----------

Requirements :
- `Docker`

Start by creating the python image : 

.. code-block:: bash

  docker build . -t python-neuralnet -f .devops/python.Dockerfile

.. note::
  We're assuming that you're running the commands from the root of the project. 
  Otherwise you'd have to adjust the paths accordingly.

Access the docker container which will act as your working environment :

.. code-block:: bash

  docker run -it python-neuralnet

To know if you're in you should see a change in the cli's prompt :

.. code-block::
  
  root@ad245b0ff5c4:/app#

Now all you have to do is navigate to the python example project of your liking and run the scripts just like you would normally.

.. code-block:: bash

  cd examples/train-predict-MNIST && python main.py

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