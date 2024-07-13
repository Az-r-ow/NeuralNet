#include "Network.hpp"

using namespace NeuralNet;

Network::Network(){};

size_t Network::getNumLayers() const { return this->layers.size(); }

void Network::setup(const std::shared_ptr<Optimizer> &optimizer, LOSS loss) {
  this->optimizer = optimizer;
  this->lossFunc = loss;
  this->setLoss(loss);
  this->updateOptimizerSetup(this->layers.size());
  this->registerSignals();  // Allows smooth exit of program
}

void Network::addLayer(std::shared_ptr<Layer> &layer) {
  size_t numLayers = this->layers.size();
  // Init layer with right amount of weights
  if (numLayers > 0) {
    const Layer &prevLayer = *this->layers[this->layers.size() - 1];
    layer->init(prevLayer.getNumNeurons());
  }

  this->layers.push_back(layer);
}

void Network::setLoss(LOSS loss) {
  switch (loss) {
    case LOSS::QUADRATIC:
      this->cmpLoss = Quadratic::cmpLoss;
      this->cmpLossGrad = Quadratic::cmpLossGrad;
      break;
    case LOSS::MCE:
      this->cmpLoss = MCE::cmpLoss;
      this->cmpLossGrad = MCE::cmpLossGrad;
      break;
    case LOSS::BCE:
      this->cmpLoss = BCE::cmpLoss;
      this->cmpLossGrad = BCE::cmpLossGrad;
      break;
    default:
      assert(false && "Loss not defined");
      break;
  }
}

std::shared_ptr<Layer> Network::getLayer(int index) const {
  assert(index < this->layers.size() && index >= 0);
  return this->layers.at(index);
}

std::shared_ptr<Layer> Network::getOutputLayer() const {
  assert(this->layers.size() > 0);
  return this->layers[this->layers.size() - 1];
}

std::string Network::getSlug() const {
  std::string slug;

  for (int l = 0; l < this->layers.size(); l++) {
    Layer &cLayer = *this->layers[l];
    slug += cLayer.getSlug() + "-";
  }

  slug.pop_back();  // remove last "-"
  return slug;
}

double Network::train(std::vector<std::vector<double>> X, std::vector<double> y,
                      int epochs,
                      std::vector<std::shared_ptr<Callback>> callbacks,
                      bool progBar) {
  this->progBar = progBar;
  try {
    return onlineTraining(X, y, epochs, callbacks);
  } catch (const std::exception &e) {
    trainingCheckpoint("onTrainEnd", callbacks);
    std::cerr << "Training Interrupted : " << e.what() << '\n';
    return loss;
  }
}

double Network::train(std::vector<std::vector<std::vector<double>>> X,
                      std::vector<double> y, int epochs,
                      std::vector<std::shared_ptr<Callback>> callbacks,
                      bool progBar) {
  this->progBar = progBar;
  try {
    return onlineTraining(X, y, epochs, callbacks);
  } catch (const std::exception &e) {
    trainingCheckpoint("onTrainEnd", callbacks);  // wrap up callbacks
    std::cerr << "Training Interrupted : " << e.what() << '\n';
    return loss;
  }
}

// Specific implementation of train that takes TrainingData class as input
double Network::train(
    TrainingData<std::vector<std::vector<double>>, std::vector<double>>
        trainingData,
    int epochs, std::vector<std::shared_ptr<Callback>> callbacks,
    bool progBar) {
  this->progBar = progBar;
  try {
    return this->trainer(trainingData, epochs, callbacks);
  } catch (const std::exception &e) {
    trainingCheckpoint("onTrainEnd", callbacks);
    std::cerr << "Training Interrupted : " << e.what() << '\n';
    return loss;
  }
}

double Network::train(
    TrainingData<std::vector<std::vector<std::vector<double>>>,
                 std::vector<double>>
        trainingData,
    int epochs, std::vector<std::shared_ptr<Callback>> callbacks,
    bool progBar) {
  this->progBar = progBar;
  try {
    return this->trainer(trainingData, epochs, callbacks);
  } catch (const std::exception &e) {
    trainingCheckpoint("onTrainEnd", callbacks);
    std::cerr << "Training Interrupted : " << e.what() << '\n';
    return loss;
  }
}

template <typename D1, typename D2>
double Network::trainer(TrainingData<D1, D2> trainingData, int epochs,
                        std::vector<std::shared_ptr<Callback>> callbacks) {
  if (trainingData.batched)
    return this->miniBatchTraining(trainingData, epochs, callbacks);
  return this->batchTraining(trainingData, epochs, callbacks);
}

template <typename D1, typename D2>
double Network::miniBatchTraining(
    TrainingData<D1, D2> trainingData, int epochs,
    std::vector<std::shared_ptr<Callback>> callbacks) {
  double sumLoss = 0;
  const int nOutputs = getOutputLayer()->getNumNeurons();
  D1 xTest = trainingData.xTest;
  D2 yTest = trainingData.yTest;

  // Formatting and caching test labels
  Eigen::MatrixXd yTestM = !yTest.empty()
                               ? formatLabels(yTest, {yTest.size(), nOutputs})
                               : Eigen::MatrixXd::Zero(1, 1);
  trainingCheckpoint("onTrainBegin", callbacks);

  // Epoch loop
  for (cEpoch = 0; cEpoch < epochs; cEpoch++) {
    double sumBatchLoss = 0;
    int nBatches = trainingData.miniBatches.size();
    trainingCheckpoint("onEpochBegin", callbacks);
    TrainingGauge g(nBatches, 0, epochs, (cEpoch + 1));

    // Batch loop
    for (int b = 0; b < nBatches; b++) {
      trainingCheckpoint("onBatchBegin", callbacks);
      auto &xTrain = trainingData.miniBatches[b].first;
      auto &yTrain = trainingData.miniBatches[b].second;
      const int nInputs = xTrain.size();
      Eigen::MatrixXd y = formatLabels(yTrain, {nInputs, nOutputs});

      // computing outputs from forward propagation
      Eigen::MatrixXd o = this->forwardProp(xTrain, true);

      loss = this->cmpLoss(o, y) / nInputs;
      accuracy = computeAccuracy(o, y);
      sumBatchLoss += loss;
      sumLoss += loss;
      this->backProp(o, y);
      trainingCheckpoint("onBatchEnd", callbacks);
      if (!this->progBar) continue;  // Skip when disabled
      g.printWithLAndA(loss, accuracy);
    }
    // predict and calculate test metrics if present
    if (!yTest.empty() && !yTestM.isZero(0)) {
      Eigen::MatrixXd oTest = this->forwardProp(xTest);
      testLoss =
          this->cmpLoss(oTest, yTestM) / static_cast<double>(xTest.size());
      testAccuracy = computeAccuracy(oTest, yTestM);
    }
    // calculating current epoch avg loss
    loss = sumBatchLoss / static_cast<double>(nBatches);
    trainingCheckpoint("onEpochEnd", callbacks);
  }

  trainingCheckpoint("onTrainEnd", callbacks);
  return loss;
}

template <typename D1, typename D2>
double Network::batchTraining(
    TrainingData<D1, D2> trainingData, int epochs,
    std::vector<std::shared_ptr<Callback>> callbacks) {
  double sumLoss = 0;
  const int nOutputs = this->getOutputLayer()->getNumNeurons();
  const int nInputs = trainingData.xTrain.size();
  D1 xTest = trainingData.xTest;
  D2 yTest = trainingData.yTest;

  // Formatting and caching test labels
  Eigen::MatrixXd yTestM = !yTest.empty()
                               ? formatLabels(yTest, {yTest.size(), nOutputs})
                               : Eigen::MatrixXd::Zero(1, 1);
  Eigen::MatrixXd y = formatLabels(trainingData.yTrain, {nInputs, nOutputs});
  trainingCheckpoint("onTrainBegin", callbacks);

  for (cEpoch = 0; cEpoch < epochs; cEpoch++) {
    trainingCheckpoint("onEpochBegin", callbacks);
    TrainingGauge g(1, 0, epochs, (cEpoch + 1));
    Eigen::MatrixXd o = this->forwardProp(trainingData.xTrain, true);

    // predict and calculate test metrics if present
    if (!yTest.empty() && !yTestM.isZero(0)) {
      Eigen::MatrixXd oTest = this->forwardProp(xTest, true);
      testLoss = this->cmpLoss(oTest, yTestM);
      testAccuracy = computeAccuracy(oTest, yTestM);
    }

    loss = this->cmpLoss(o, y) / nInputs;
    accuracy = computeAccuracy(o, y);
    sumLoss += loss;

    this->backProp(o, y);
    trainingCheckpoint("onEpochEnd", callbacks);
    if (!this->progBar) continue;  // Skip when disabled
    g.printWithLAndA(loss, accuracy);
  }

  trainingCheckpoint("onTrainEnd", callbacks);
  return sumLoss / nInputs;
}

template <typename D1, typename D2>
double Network::onlineTraining(
    std::vector<D1> inputs, std::vector<D2> labels, int epochs,
    std::vector<std::shared_ptr<Callback>> callbacks) {
  double sumLoss = 0;
  int tCorrect = 0;
  const int numOutputs = this->getOutputLayer()->getNumNeurons();
  const int numInputs = inputs.size();
  Eigen::MatrixXd y = formatLabels(labels, {numInputs, numOutputs});

  // Injecting callbacks
  trainingCheckpoint("onTrainBegin", callbacks);

  for (cEpoch = 0; cEpoch < epochs; cEpoch++) {
    trainingCheckpoint("onEpochBegin", callbacks);
    TrainingGauge tg(inputs.size(), 0, epochs, (cEpoch + 1));
    for (auto &input : inputs) {
      Eigen::MatrixXd o = this->forwardProp(inputs, true);
      loss = this->cmpLoss(o, y);
      sumLoss += loss;
      tCorrect += computeAccuracy(o, y);
      this->backProp(o, y);
      if (!this->progBar) continue;  // Skip when disabled
      tg.printWithLoss(loss);
    }
    // Computing metrics for the logs
    accuracy = tCorrect / numInputs;
    loss = sumLoss / numInputs;
    trainingCheckpoint("onEpochEnd", callbacks);
  }

  trainingCheckpoint("onTrainEnd", callbacks);
  return sumLoss / numInputs;
}

Eigen::MatrixXd Network::predict(std::vector<std::vector<double>> inputs) {
  Eigen::MatrixXd mInputs = vectorToMatrixXd(inputs);
  return forwardProp(mInputs);
}

Eigen::MatrixXd Network::predict(
    std::vector<std::vector<std::vector<double>>> inputs) {
  return forwardProp(inputs);
}

/**
 * Forward propagation
 */
Eigen::MatrixXd Network::feedForward(Eigen::MatrixXd inputs, int startIdx,
                                     bool training) {
  assert(startIdx < this->layers.size());
  Eigen::MatrixXd prevLayerOutputs = inputs;

  for (int l = startIdx; l < this->layers.size(); l++) {
    Layer &cLayer = *this->layers[l];
    if (cLayer.trainingOnly && !training) continue;
    prevLayerOutputs = cLayer.feedInputs(prevLayerOutputs, training);
  }

  return prevLayerOutputs;
}

Eigen::MatrixXd Network::forwardProp(
    std::vector<std::vector<std::vector<double>>> &inputs, bool training) {
  // Passing the inputs as outputs to the input layer
  this->layers[0]->feedInputs(inputs);

  Eigen::MatrixXd prevLayerOutputs = this->layers[0]->getOutputs();

  return feedForward(prevLayerOutputs, 1, training);
}

Eigen::MatrixXd Network::forwardProp(std::vector<std::vector<double>> &inputs,
                                     bool training) {
  // Previous layer outputs
  Eigen::MatrixXd prevLayerO = vectorToMatrixXd(inputs);

  return feedForward(prevLayerO, 0, training);
}

Eigen::MatrixXd Network::forwardProp(Eigen::MatrixXd &inputs, bool training) {
  // Previous layer outputs
  Eigen::MatrixXd prevLayerO = inputs;

  return feedForward(prevLayerO, 0, training);
}

void Network::backProp(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y) {
  // Next Layer activation der dL/da(l - 1)
  Eigen::MatrixXd beta = this->cmpLossGrad(outputs, y);
  int m = beta.rows();

  for (size_t i = this->layers.size(); --i > 0;) {
    Layer &cLayer = *this->layers[i];
    Layer &nLayer = *this->layers[i - 1];

    Eigen::MatrixXd nLayerOutputs = nLayer.getOutputs();

    Dense *cDense = dynamic_cast<Dense *>(&cLayer);

    if (!cDense || !nLayerOutputs.cols() || !nLayerOutputs.rows()) continue;

    if (nLayer.type == LayerType::DROPOUT) {
      // dropout layer
      Dropout *doLayer = dynamic_cast<Dropout *>(&nLayer);

      assert(doLayer);
      // rescale outputs
      nLayerOutputs = nLayerOutputs.array() * doLayer->mask.array();
      nLayerOutputs /= doLayer->scaleRate;
    }

    // a'(L)
    Eigen::MatrixXd aDer = cDense->diff(cDense->outputs);

    // a(L - 1) . a'(L)
    Eigen::MatrixXd delta = beta.array() * aDer.array();

    Eigen::MatrixXd gradW = (1.0 / m) * (nLayerOutputs.transpose() * delta);

    Eigen::MatrixXd gradB = (1.0 / m) * delta.colwise().sum();

    // dL/dA(l - 1)
    beta = delta * cDense->weights.transpose();

    // updating weights and biases
    this->optimizer->updateWeights(cDense->weights, gradW);
    this->optimizer->updateBiases(cDense->biases, gradB);
  }
}

void Network::updateOptimizerSetup(size_t numLayers) {
  /**
   * This is a way to let adams know about the number of layers
   * With that it can setup the `l` variable and the std::vectors
   *
   * I'm not very proud of this method but so far it seems like the most
   * convenient way
   *
   * Adam only requires the number of Dense layers
   */

  size_t nLayers = numLayers;

  if (std::dynamic_pointer_cast<Adam>(this->optimizer)) {
    // Get the number of dense layers
    nLayers = std::count_if(
        this->layers.begin(), this->layers.end(),
        [](const std::shared_ptr<Layer> &ptr) {
          return std::dynamic_pointer_cast<Dense>(ptr) != nullptr ||
                 std::dynamic_pointer_cast<Flatten>(ptr) != nullptr;
        });
  }

  this->optimizer->insiderInit(nLayers);
}

void Network::trainingCheckpoint(
    std::string checkpointName,
    std::vector<std::shared_ptr<Callback>> callbacks) {
  if (callbacks.size() == 0) return;

  for (std::shared_ptr<Callback> callback : callbacks) {
    Callback::callMethod(callback, checkpointName, *this);
  }
}

/**
 * @note This function will return the accuracy of the given outputs compared to
 * the labels.
 *
 * @param outputs The outputs from the network
 * @param y The labels
 *
 * @return The accuracy of the network.
 */
double Network::computeAccuracy(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y) {
  int total = y.rows();

  // Hardmax the outputs
  Eigen::MatrixXd outputsHm = hardmax(outputs);

  Eigen::MatrixXd diff = outputsHm - y;

  int wrong = diff.cwiseAbs().sum() / 2;

  return 1.0 - (wrong / static_cast<double>(total));
}

Network::~Network() {}