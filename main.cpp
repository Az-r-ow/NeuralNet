#include "main.hpp"

using namespace NeuralNet;

int main(int argc, char *argv[])
{

  Eigen::MatrixXd m(2, 2);

  m << -4, 0,
      -1, -4;

  Eigen::MatrixXd m2 = m.cwiseAbs();

  std::cout << m2 << "\n";
  // Network network;
  // std::shared_ptr<Optimizer> AdamOptimizer = std::make_shared<Adam>(1);

  // std::shared_ptr<Layer> layer1 = std::make_shared<Dense>(3, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);
  // std::shared_ptr<Layer> layer2 = std::make_shared<Dense>(2, ACTIVATION::SIGMOID, WEIGHT_INIT::HE);
  // std::shared_ptr<Layer> layerOuput = std::make_shared<Dense>(2, ACTIVATION::SIGMOID, WEIGHT_INIT::GLOROT);

  // network.addLayer(layer1);
  // network.addLayer(layer2);
  // network.addLayer(layerOuput);

  // std::shared_ptr<Layer> l = network.getLayer(1);
  // std::cout << "fetched layer from network : " << l->getNumNeurons() << "\n";
  // network.setup(AdamOptimizer);

  // network.setup(AdamOptimizer, LOSS::QUADRATIC);

  // std::cout << "num of layers : " << network.getNumLayers() << "\n";

  // std::cout
  //     << "Input Dense before training : "
  //     << "\n";
  // layer1->printWeights();
  // layer1->printOutputs();

  // std::cout << "Dense 2 before training : "
  //           << "\n";
  // layer2->printWeights();
  // layer2->printOutputs();

  // std::cout << "Output Dense before training : "
  //           << "\n";
  // layerOuput->printWeights();
  // layerOuput->printOutputs();

  // // training the network
  // std::vector<std::vector<double>> inputs;
  // inputs.push_back(randDVector(layer1->getNumNeurons(), -1, 1));
  // inputs.push_back(randDVector(layer1->getNumNeurons(), -1, 1));
  // std::vector<double> labels = {1, 1};

  // TrainingData tr_data(inputs, labels);
  // tr_data.batch(1);

  // network.train(tr_data);

  // std::shared_ptr<Layer> input = network.getLayer(0);
  // std::shared_ptr<Layer> test = network.getLayer(1);
  // std::shared_ptr<Layer> test2 = network.getLayer(2);

  // std::cout << "Input Dense after training : "
  //           << "\n";
  // input->printWeights();
  // input->printOutputs();

  // std::cout << "Dense 2 after training : "
  //           << "\n";
  // test->printWeights();
  // test->printOutputs();

  // std::cout << "Output Dense after training : "
  //           << "\n";
  // test2->printWeights();
  // test2->printOutputs();

  // std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8};

  // Tensor t(data);

  // t.batch(2);

  // std::vector<std::vector<double>> batch = t.getBatchedData();

  // std::cout << "Num batches : " << batch.size() << "\n";

  // TrainingData td(data, data);

  // td.batch(2);
}