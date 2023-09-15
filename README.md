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

## Activation functions and Scaling :

For the **Softmax** activation I had to scale the inputs because of how delicate the function is. Too high and I get an `inf` and too low and I'd kill neurons.
So I would scale all the inputs based on the inverse of the highest number.

$$
\frac{1}{max(i)}
$$

In the case where the highest number was big the inputs would be scaled **down** and in the case where the highest number is small the inputs would be scaled **up** accordingly.

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
