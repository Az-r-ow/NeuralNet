# Neural Network in CPP :

This is my take on implementing a neural network in cpp. Keeping in mind that I learned cpp a couple of weeks before I started the project.

## The importance of weight initialization functions

Arbitrary initialization can slow down and sometimes stall completely the convergence process. This slowdown can result in the deeper layers receiving inputs with small variances, which in turn slows down back propagation, and retards the overall convergence progress.
