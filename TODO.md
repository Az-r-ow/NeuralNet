## TODOS :

- [ ] CI versioning
- [ ] Implement batch norm
- [ ] Parallelize operations
- [ ] Read : https://arxiv.org/pdf/1412.6980.pdf
- [ ] Implement a dropout to avoid over-fitting the model
- [ ] Add macos arm runner when available

## IN PROGRESS :

- [ ] Python tests
- [ ] Optimize `Catch2`'s build
- [ ] Add gradient clipping

## DONE :

- [x] Setup `clang-format`
- [x] Implement early stopping
- [x] Update README to include more information about the project
- [x] Add CI / CD
- [x] Document the example in python
- [x] Re-implement the whole mini-batch algorithm
- [x] Comment the code further
- [x] Find out if adam optimization is working
- [x] Add type for data handling
- [x] Adams optimizer
- [x] Read : https://arxiv.org/pdf/1812.03372.pdf
- [x] Fix the he weight initialisation
- [x] added License
- [x] Replace Quadratic loss der in lossDer function with dynamic function based on loss used
- [x] Change the enums names to all capital (ex: ACTIVATION, LOSS, WEIGHT_INIT)
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
  - Reason : Dig deeper into it, at first thought it might not be worth switching to header guards. Pragma once is supported by most compilers and it avoids name clashes in namespace.
