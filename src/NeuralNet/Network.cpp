#include "Network.hpp"

using namespace NeuralNet;

Network::Network(double alpha) : alpha(alpha){};

size_t Network::getNumLayers() const
{
  return this->layers.size();
}

void Network::setup(const std::shared_ptr<Optimizer> &optimizer, int epochs, LOSS loss)
{
  this->optimizer = optimizer;
  this->epochs = epochs;
  this->lossFunc = loss;
  this->setLoss(loss);
  this->updateOptimizerSetup(this->layers.size());
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

double Network::train(std::vector<std::vector<double>> inputs, std::vector<double> labels)
{
  return onelineTraining(inputs, labels);
}

double Network::train(std::vector<std::vector<std::vector<double>>> inputs, std::vector<double> labels)
{
  return onelineTraining(inputs, labels);
}

// Specific implementation of train that takes TrainingData class as input
double Network::train(TrainingData<std::vector<std::vector<double>>, std::vector<double>> trainingData)
{
  return this->trainer(trainingData);
}

double Network::train(TrainingData<std::vector<std::vector<std::vector<double>>>, std::vector<double>> trainingData)
{
  return this->trainer(trainingData);
}

template <typename D1, typename D2>
double Network::trainer(TrainingData<D1, D2> trainingData)
{
  if (trainingData.batched)
    return this->miniBatchTraining(trainingData);
  return this->batchTraining(trainingData);
}

template <typename D1, typename D2>
double Network::miniBatchTraining(TrainingData<D1, D2> trainingData)
{
  double loss;

  for (int e = 0; e < epochs; e++)
  {
    TrainingGauge g(trainingData.inputs.size(), 0, epochs, (e + 1));
    for (int b = 0; b < trainingData.inputs.size(); b++)
    {
      const int numOutputs = this->getOutputLayer()->getNumNeurons();
      const int inputsSize = trainingData.inputs.batches[b].size();
      Eigen::MatrixXd y = formatLabels(trainingData.labels.batches[b], {inputsSize, numOutputs});
      Eigen::MatrixXd lossGrad = zeroMatrix({inputsSize, numOutputs});

      // computing outputs from forward propagation
      Eigen::MatrixXd o = this->forwardProp(trainingData.inputs.batches[b]);
      loss = this->cmpLoss(o, y) / inputsSize;
      this->backProp(o, y);
      g.printWithLoss(loss);
    }
  }

  return loss;
}

template <typename D1, typename D2>
double Network::batchTraining(TrainingData<D1, D2> trainingData)
{
  double loss;
  const int numOutputs = this->getOutputLayer()->getNumNeurons();
  const int inputsSize = trainingData.inputs.data.size();
  Eigen::MatrixXd y = formatLabels(trainingData.labels.data, {inputsSize, numOutputs});

  TrainingGauge g(1, 0, epochs, (e + 1));
  for (int e = 0; e < epochs; e++)
  {
    Eigen::MatrixXd o = this->forwardProp(trainingData.inputs.data);

    loss = this->cmpLoss(o, y);

    this->backProp(o, y);
    g.printWithLoss(loss);
  }

  return loss;
}

template <typename D1, typename D2>
double Network::onelineTraining(std::vector<D1> inputs, std::vector<D2> labels)
{
  double loss;
  const int numOutputs = this->getOutputLayer()->getNumNeurons();
  const int inputsSize = inputs.size();
  Eigen::MatrixXd y = formatLabels(labels, {inputsSize, numOutputs});

  for (int e = 0; e < epochs; e++)
  {
    Eigen::MatrixXd o = this->forwardProp(inputs);

    loss = this->cmpLoss(o, y);

    this->backProp(o, y);
  }

  return loss;
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
 * Updates the correct prediction and the total prediction
 * Returns the accuracy of the model
 */
double Network::computeAccuracy(const int predicted, const int label)
{
  this->tp += 1;

  if (predicted == label)
  {
    this->cp += 1;
  }

  return static_cast<double>(cp) / tp;
}

Network::~Network()
{
}