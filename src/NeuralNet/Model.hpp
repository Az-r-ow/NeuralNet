#pragma once

#include <string>

namespace NeuralNet
{
  class Model
  {
  public:
    void save(const std::string &filename);
    static Model load(const std::string &filename);
  };
}