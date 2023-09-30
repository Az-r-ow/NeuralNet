## TODOS :

- [ ] Check what's wrong with loaded model not training
- [ ] Read : https://arxiv.org/pdf/1412.6980.pdf
- [ ] Comment the code further
- [ ] Implement a dropout to avoid over-fitting the model

## IN PROGRESS :

- [ ] Update README to include more information about the project

## DONE :

- [x] Adams optimizer
- [x] Read : https://arxiv.org/pdf/1812.03372.pdf
- [x] Fix the he weight initialisation
- [x] added License
- [x] Replace Quadratic loss der in lossDer function with dynamic function based on loss used
- [x] Change the enums names to all capital (ex: ACTIVATION, LOSS, WEIGHT_INIT)
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
