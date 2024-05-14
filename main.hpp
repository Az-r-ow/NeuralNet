#pragma once

#include <cereal/archives/binary.hpp>
#include <fstream>
#include <iostream>

#include "src/NeuralNet/Network.hpp"
#include "src/NeuralNet/data/Tensor.hpp"
#include "src/NeuralNet/layers/Dense.hpp"
#include "src/NeuralNet/layers/Layer.hpp"
#include "src/NeuralNet/optimizers/optimizers.hpp"
#include "src/NeuralNet/utils/Functions.hpp"