## TODOS :

- [ ] Change the enums names to all capital (ex: ACTIVATION, LOSS, WEIGHT_INIT)
- [ ] Read : https://arxiv.org/pdf/1812.03372.pdf
- [ ] Implement a dropout to avoid over-fitting the model
- [ ] Implement the ability of having multiple variable inputs

## IN PROGRESS :

- [ ] Replace Quadratic loss der in lossDer function with dynamic function based on loss used
  - [ ] Implement more cost functions options

## DONE :

- [x] Implement mini-batch
- [x] Test with the MNIST database
- [x] Test the backpropagation
- [x] Implement the sigmoid class
- [x] Write some unit tests with the Catch2 library
- [x] Transform labels into correct format for evaluation
- [x] Connect with python using pybind

## ARCHIVED :

- [x] Work on optimizing the way files are including one another
- [ ] Add cmake option to install submodules automatically during build time
- [ ] Replace pragma once with header guards
  - Reason : Look more into it, at first glance it might not be worth switching to header guards. Pragma once is supported by most compilers and it avoids name clashes in namespace.
