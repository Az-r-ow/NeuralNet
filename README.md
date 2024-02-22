# NeuralNet in CPP :

This is my take on implementing a neural network in cpp. Keeping in mind that I learned cpp a couple of weeks prior to starting the project. You can see my programming style adapting and improving (hopefully) throughout the commits.

- [NeuralNet in CPP :](#neuralnet-in-cpp-)
  - [Build](#build)
    - [Initialize submodules](#initialize-submodules)
    - [Build the code](#build-the-code)
  - [Tests](#tests)
  - [📖 Docs](#-docs)
  - [Miscellaneous](#miscellaneous)
    - [🔗 Python Bindings](#-python-bindings)
    - [The importance of weight initialization functions](#the-importance-of-weight-initialization-functions)
      - [Available Weight Initializations](#available-weight-initializations)
  - [⚖️ License](#️-license)

## Build

### Initialize submodules

```bash
git submodule init
git submodule update
```

### Build the code

```bash
scripts/build.sh
```

## Tests

[Catch 2](https://github.com/catchorg/Catch2) framework will be used for testing, after some research it seems like the most active and well maintained out of the other options.

To run tests :

```bash
source /scripts/tests.sh
```

## 📖 Docs

- [cpp docs 📖](https://az-r-ow.github.io/NeuralNet/)
- python docs (coming soon...)

## Miscellaneous

### 🔗 Python Bindings

I used the [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) library to bind some of the classes and functionalities. After [building](#build) the project you can head to `/examples` folder to check out some of the cool mini-projects built in python.

### The importance of weight initialization functions

Arbitrary initialization can slow down and sometimes stall completely the convergence process. This slowdown can result in the deeper layers receiving inputs with small variances, which in turn slows down back propagation, and slows down the overall convergence progress.

#### Available Weight Initializations

| WEIGHT_INIT | Formula                      | Activation      |
| ----------- | ---------------------------- | --------------- |
| RANDOM      | $mtRand(-1, 1)$              | Sigmoid         |
| GLOROT      | $\frac{2}{n_{in} + n_{out}}$ | Relu            |
| HE          | $\frac{2}{n_{in}}$           | Relu<br>Softmax |
| LECUN       | $\frac{1}{n_{in}}$           | Softmax         |

$n_{in}$ number of inputs

$n_{out}$ number of outputs

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
