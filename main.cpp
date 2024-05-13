#include "main.hpp"

#include <iostream>
#include <string>

using namespace NeuralNet;

int main(int argc, char *argv[]) {
  std::string fileName = "test.cpp";
  std::string folderPath = "build/";

  std::string filepath = constructFilePath(folderPath, fileName);

  std::cout << "Constructed file path : " << filepath << "\n";

  Network test;

  std::unique_ptr<Network> ptr = std::make_unique<Network>();

  decltype(*ptr) obj = *ptr;

  std::cout << "Type of obj: " << typeid(obj).name() << std::endl;
}