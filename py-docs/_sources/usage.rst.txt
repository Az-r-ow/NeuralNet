Usage
=====

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