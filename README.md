# Neural Network in CPP :

This is my take on implementing a neural network in cpp. Keeping in mind that I learned cpp a couple of weeks before I started the project.

## The importance of weight initialization functions

Arbitrary initialization can slow down and sometimes stall completely the convergence process. This slowdown can result in the deeper layers receiving inputs with small variances, which in turn slows down back propagation, and retards the overall convergence progress.

### Available Weight Initializations :

| WEIGHT_INIT | Formula                      | Activation       |
| ----------- | ---------------------------- | ---------------- |
| RANDOM      | $mtRand(-1, 1)$              | Sigmoid          |
| GLOROT      | $\frac{2}{n_{in} + n_{out}}$ | Relu             |
| HE          | $\frac{2}{n_{in}}$           | Relu <br>Softmax |
| LECUN       | $\frac{1}{n_{in}}$           | Softmax          |

$n_{in}$ number of inputs

$n_{out}$ number of outputs

## Activation functions and Scaling :

### Sigmoid :

The sigmoid function, denoted as σ(z), transforms a real number 'z' into a value between 0 and 1. It's characterized by an S-shaped curve and is commonly used in binary classification tasks, where it models probabilities. For example, σ(z) = 0.7 indicates a 70% probability of belonging to one class.

$$
\sigma \left(x \right) = \frac{1}{1 + e^{-x}}
$$

![Sigmoid Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

### Relu :

The Rectified Linear Unit (ReLU) computes the output as follows: If the input is positive, it returns the input value; if negative, it returns 0. ReLU is computationally efficient, helps mitigate the vanishing gradient problem, and is widely used in hidden layers of deep networks for tasks like image recognition and natural language processing. Hence, why I'm using it in my MNIST example in python.

![RELU vs GELU](https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/220px-ReLU_and_GELU.svg.png)

### Softmax :

The softmax function is a versatile activation used mainly in the output layer of neural networks for multi-class classification tasks. It takes a vector of real numbers as input and transforms them into a probability distribution over multiple classes, ensuring that the output values sum up to 1. It's known for its smooth, differentiable properties and is particularly useful for scenarios where you need to model class probabilities. For example, in a 3-class problem, softmax converts input scores into probabilities like [0.2, 0.7, 0.1], indicating a 70% chance of belonging to class 2.
I also used for the output layer of my MNIST example.

$$
\sigma \left(z\right)_{i} = \frac{e^{z_{i}}}{\sum_{j=1}^{k} e^{z_{j}}}
$$

## Tests :

[Catch 2](https://github.com/catchorg/Catch2) framework will be used for testing, after some research it seems like the most active and well maintained out of the other options.

To run tests :

```bash
source /scripts/tests.sh
```

## Build :

### Initialize submodules :

```bash
git submodule init
git submodule update
```

### Build the code :

```bash
./scripts/build.sh
```

### License :

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
