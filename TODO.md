## TODOS :

- [ ] Document the example in python
- [ ] Find out why the predictions are not accurate on my_samples
- [ ] Find out if adam optimization is working
- [ ] Read : https://arxiv.org/pdf/1412.6980.pdf
- [ ] Comment the code further
- [ ] Implement a dropout to avoid over-fitting the model

## IN PROGRESS :

- [ ] Add types that handle Batch - Mini-batch - Oneline
- [ ] Add gradient clipping
- [ ] Re-implement the whole mini-batch algorithm
  - [x] Find a way to bind methods that have overrides
  - [x] Make sure the whole shared_ptr layers don't affect serialization
  - [x] Make Layer a base class and add Dense Layer to replace Layer
  - [x] Check what's up with the main python script and the reason why the shapes aren't matching
    - [x] Change the biases into vectors (this is causing the unmatching shapes error)
    - [x] Adapt the return value of the predict function
    - [x] Adapt predict function for flatten layer
    - [x] Adapt Dense layer for serialization
    - [ ] Fix the loading bar
      - [x] Fix the loading bar for mini-batch training
      - [ ] Fix the loading bar for online training
- [ ] Update README to include more information about the project

## DONE :

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
