#pragma once

#include <variant>

#include "Model.hpp"

namespace NeuralNet {
using Logs = std::variant<double, int>;
}  // namespace NeuralNet