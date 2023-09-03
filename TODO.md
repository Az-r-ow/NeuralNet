## TODOS :

- [ ] Replace pragma once with header guards
- [ ] Read : https://arxiv.org/pdf/1812.03372.pdf
- [ ] Replace Quadratic loss der in lossDer function with dynamic function based on loss used
- [ ] Implement mini-batch
- [ ] Implement more cost functions options
- [ ] Implement the ability of having multiple variable inputs

## IN PROGRESS :

- [ ] Test with the MNIST database
  - [x] Introduce ftxui to the project for feedback while training
  - [x] Calculate accuracy during training and output it
  - [x] Remove training string from TrainingGauge constructor
  - [ ] Fix the test cases

## DONE :

- [x] Test the backpropagation
- [x] Implement the sigmoid class
- [x] Write some unit tests with the Catch2 library
- [x] Transform labels into correct format for evaluation
- [x] Connect with python using pybind

## ARCHIVED :

- [x] Work on optimizing the way files are including one another
- [ ] Add cmake option to install submodules automatically during build time
