#include "Network.hpp"

using namespace NeuralNet;

Network::Network(){};

size_t Network::getNumLayers() const
{
  return this->layers.size();
}

void Network::setup(const std::shared_ptr<Optimizer> &optimizer, LOSS loss)
{
  this->optimizer = optimizer;
  this->lossFunc = loss;
  this->setLoss(loss);
  this->updateOptimizerSetup(this->layers.size());
  this->registerSignals(); // Allows smooth exit of program
}

void Network::addLayer(std::shared_ptr<Layer> &layer)
{
  size_t numLayers = this->layers.size();
  // Init layer with right amount of weights
  if (numLayers > 0)
  {
    std::shared_ptr<Layer> prevLayer = this->layers[this->layers.size() - 1];
    layer->init(prevLayer->getNumNeurons());
  }

  this->layers.push_back(layer);
}

void Network::setLoss(LOSS loss)
{
  switch (loss)
  {
  case LOSS::QUADRATIC:
    this->cmpLoss = Quadratic::cmpLoss;
    this->cmpLossGrad = Quadratic::cmpLossGrad;
    break;
  case LOSS::MCE:
    this->cmpLoss = MCE::cmpLoss;
    this->cmpLossGrad = MCE::cmpLossGrad;
    break;
  default:
    assert(false && "Loss not defined");
    break;
  }
}

std::shared_ptr<Layer> Network::getLayer(int index) const
{
  assert(index < this->layers.size() && index >= 0);
  return this->layers.at(index);
}

std::shared_ptr<Layer> Network::getOutputLayer() const
{
  assert(this->layers.size() > 0);
  return this->layers[this->layers.size() - 1];
}

double Network::train(std::vector<std::vector<double>> inputs, std::vector<double> labels, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  return onlineTraining(inputs, labels, epochs);
}

double Network::train(std::vector<std::vector<std::vector<double>>> inputs, std::vector<double> labels, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  return onlineTraining(inputs, labels, epochs);
}

// Specific implementation of train that takes TrainingData class as input
double Network::train(TrainingData<std::vector<std::vector<double>>, std::vector<double>> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  return this->trainer(trainingData, epochs);
}

double Network::train(TrainingData<std::vector<std::vector<std::vector<double>>>, std::vector<double>> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  return this->trainer(trainingData, epochs);
}

template <typename D1, typename D2>
double Network::trainer(TrainingData<D1, D2> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  if (trainingData.batched)
    return this->miniBatchTraining(trainingData, epochs, callbacks);
  return this->batchTraining(trainingData, epochs, callbacks);
}

template <typename D1, typename D2>
double Network::miniBatchTraining(TrainingData<D1, D2> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  double loss;
  double sumLoss = 0;

  for (int e = 0; e < epochs; e++)
  {
    TrainingGauge g(trainingData.inputs.size(), 0, epochs, (e + 1));
    for (int b = 0; b < trainingData.inputs.size(); b++)
    {
      const int numOutputs = this->getOutputLayer()->getNumNeurons();
      const int inputsSize = trainingData.inputs.batches[b].size();
      Eigen::MatrixXd y = formatLabels(trainingData.labels.batches[b], {inputsSize, numOutputs});

      // computing outputs from forward propagation
      Eigen::MatrixXd o = this->forwardProp(trainingData.inputs.batches[b]);
      loss = this->cmpLoss(o, y) / inputsSize;
      sumLoss += loss;
      this->backProp(o, y);
      g.printWithLAndA(loss, computeAccuracy(o, y));
    }
  }

  return sumLoss / trainingData.inputs.size();
}

template <typename D1, typename D2>
double Network::batchTraining(TrainingData<D1, D2> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  double loss;
  double sumLoss = 0;
  const int numOutputs = this->getOutputLayer()->getNumNeurons();
  const int numInputs = trainingData.inputs.data.size();
  Eigen::MatrixXd y = formatLabels(trainingData.labels.data, {numInputs, numOutputs});

  for (int e = 0; e < epochs; e++)
  {
    TrainingGauge g(1, 0, epochs, (e + 1));
    Eigen::MatrixXd o = this->forwardProp(trainingData.inputs.data);

    loss = this->cmpLoss(o, y);
    sumLoss += loss;

    this->backProp(o, y);
    g.printWithLoss(loss);
  }

  return sumLoss / numInputs;
}

template <typename D1, typename D2>
double Network::onlineTraining(std::vector<D1> inputs, std::vector<D2> labels, int epochs, std::vector<std::shared_ptr<Callback>> callbacks)
{
  double loss;
  double sumLoss;
  const int numOutputs = this->getOutputLayer()->getNumNeurons();
  const int numInputs = inputs.size();
  Eigen::MatrixXd y = formatLabels(labels, {numInputs, numOutputs});

  for (int e = 0; e < epochs; e++)
  {
    TrainingGauge tg(inputs.size(), 0, epochs, (e + 1));
    for (auto &input : inputs)
    {
      Eigen::MatrixXd o = this->forwardProp(inputs);
      loss = this->cmpLoss(o, y);
      sumLoss += loss;
      this->backProp(o, y);
      tg.printWithLoss(loss);
    }
  }

  return sumLoss / numInputs;
}

Eigen::MatrixXd Network::predict(std::vector<std::vector<double>> inputs)
{
  Eigen::MatrixXd mInputs = vectorToMatrixXd(inputs);
  return forwardProp(mInputs);
}

Eigen::MatrixXd Network::predict(std::vector<std::vector<std::vector<double>>> inputs)
{
  return forwardProp(inputs);
}

/**
 * Forward propagation
 */
Eigen::MatrixXd Network::feedForward(Eigen::MatrixXd inputs, int startIdx)
{
  assert(startIdx < this->layers.size());
  Eigen::MatrixXd prevLayerOutputs = inputs;

  for (int l = startIdx; l < this->layers.size(); l++)
  {
    prevLayerOutputs = this->layers[l]->feedInputs(prevLayerOutputs);
  }

  return prevLayerOutputs;
}

Eigen::MatrixXd Network::forwardProp(std::vector<std::vector<std::vector<double>>> &inputs)
{
  // Passing the inputs as outputs to the input layer
  this->layers[0]->feedInputs(inputs);

  Eigen::MatrixXd prevLayerOutputs = this->layers[0]->getOutputs();

  return feedForward(prevLayerOutputs, 1);
}

Eigen::MatrixXd Network::forwardProp(std::vector<std::vector<double>> &inputs)
{
  // Previous layer outputs
  Eigen::MatrixXd prevLayerO = vectorToMatrixXd(inputs);

  return feedForward(prevLayerO);
}

Eigen::MatrixXd Network::forwardProp(Eigen::MatrixXd &inputs)
{
  // Previous layer outputs
  Eigen::MatrixXd prevLayerO = inputs;

  return feedForward(prevLayerO);
}

void Network::backProp(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y)
{
  // Next Layer activation der dL/da(l - 1)
  Eigen::MatrixXd beta = this->cmpLossGrad(outputs, y);
  int m = beta.rows();

  for (size_t i = this->layers.size(); --i > 0;)
  {
    std::shared_ptr<Layer> cLayer = this->layers[i];
    std::shared_ptr<Layer> nLayer = this->layers[i - 1];
    // a'(L)
    Eigen::MatrixXd aDer = cLayer->diff(cLayer->outputs);

    // a(L - 1) . a'(L)
    Eigen::MatrixXd delta = beta.array() * aDer.array();

    Eigen::MatrixXd gradW = (1.0 / m) * (nLayer->outputs.transpose() * delta);

    Eigen::MatrixXd gradB = (1.0 / m) * delta.colwise().sum();

    // dL/dA(l - 1)
    beta = delta * cLayer->weights.transpose();

    // updating weights and biases
    this->optimizer->updateWeights(cLayer->weights, gradW);
    this->optimizer->updateBiases(cLayer->biases, gradB);
  }
}

void Network::updateOptimizerSetup(size_t numLayers)
{
  /**
   * This is a way to let adams know about the number of layers
   * With that it can setup the `l` variable and the std::vectors
   *
   * I'm not very proud of this method but so far it seems like the most convenient way
   */
  this->optimizer->insiderInit(numLayers);
}

/**
 * @note This function will return the accuracy of the given outputs compared to the labels.
 *
 * @param outputs The outputs from the network
 * @param y The labels
 *
 * @return The accuracy of the network.
 */
double Network::computeAccuracy(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y)
{
  int total = y.rows();

  // Hardmax the outputs
  Eigen::MatrixXd outputsHm = hardmax(outputs);

  Eigen::MatrixXd diff = outputsHm - y;

  int wrong = diff.cwiseAbs().sum() / 2;

  return 1.0 - (wrong / static_cast<double>(total));
}

Network::~Network()
{
}