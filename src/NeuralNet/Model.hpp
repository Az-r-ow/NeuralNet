#pragma once

#include <string>
#include <fstream>
#include <type_traits>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include "utils/Functions.hpp"

namespace NeuralNet
{
  class Model
  {
  public:
    /**
     * @brief This method will save (by serializing) the model passed as argument to a .bin file
     *
     * @param filename The binary file in which the serialized model will be saved
     * @param model The model that will be serialized and saved
     */
    template <typename T, typename = typename std::enable_if<std::is_base_of<Model, T>::value>::type>
    static void save_to_file(const std::string &filename, T model)
    {
      // Serializing model to a binary file
      std::ofstream file(filename, std::ios::binary);
      cereal::BinaryOutputArchive archive(file);
      archive(model);
    };

    /**
     * @brief This static method loads a Model from a file and assigns it to the supposedly "empty" model passed as argument
     *
     * @param filename The name of the binary file from which to retrieve the model
     * @param model An "empty" model which will be filled with the loaded model's parameters
     *
     * This function will assign the parameters of the saved model in the binary file to the model passed as a parameter
     */
    template <typename T, typename = typename std::enable_if<std::is_base_of<Model, T>::value>::type>
    static void
    load_from_file(const std::string &filename, T &model)
    {
      // Making sure the file exists and is binary
      assert(fileExistsWithExtension(filename, ".bin") && "The file doesn't exists or is not binary '.bin'");

      // Deserializing the model from the binary file
      std::ifstream file(filename, std::ios::binary);
      cereal::BinaryInputArchive archive(file);
      archive(model);
    };
  };
}