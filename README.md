# Neural Network in CPP :

This is my take on implementing a neural network in cpp. Keeping in mind that I learned cpp a couple of weeks before I started the project.

## The importance of weight initialization functions

Arbitrary initialization can slow down and sometimes stall completely the convergence process. This slowdown can result in the deeper layers receiving inputs with small variances, which in turn slows down back propagation, and retards the overall convergence progress.

## Tests :

[Catch 2](https://github.com/catchorg/Catch2) framework will be used for testing, after some research it seems like the most active and well maintained out of the other options.

To run tests :

```bash
/scripts/tests.sh
```

## Build :

### Initialize submodules :

```bash
git submodule init
git submodule update
```
