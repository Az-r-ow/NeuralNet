#pragma once

#include <cereal/cereal.hpp>  // for defer
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cstdlib>
#include <memory>
#include <variant>
#include <vector>

#include "Model.hpp"
#include "callbacks/Callback.hpp"
#include "data/TrainingData.hpp"
#include "layers/Dense.hpp"
#include "layers/Dropout.hpp"
#include "layers/Flatten.hpp"
#include "layers/Layer.hpp"
#include "losses/losses.hpp"
#include "optimizers/Optimizer.hpp"
#include "optimizers/optimizers.hpp"
#include "utils/Formatters.hpp"
#include "utils/Functions.hpp"
#include "utils/Gauge.hpp"
#include "utils/Variants.hpp"

namespace NeuralNet {
class Layer;

class Network : public Model {
 public:
  Network();

  /**
   * @brief Method that sets up the model's hyperparameter
   *
   * @param optimizer An Optimizer's child class
   * @param epochs The number of epochs
   * @param loss The loss function
   */
  void setup(const std::shared_ptr<Optimizer> &optimizer,
             LOSS loss = LOSS::QUADRATIC);

  /**
   * @brief Method to add a layer to the network
   *
   * @param layer the layer to add to the model it should be of type Layer
   */
  void addLayer(std::shared_ptr<Layer> &layer);

  /**
   * @brief This method will set the network's loss function
   *
   * @param loss The loss function (choose from the list of LOSS enums)
   */
  void setLoss(LOSS loss);

  /**
   * @brief This method will return the Layer residing at the specified index
   *
   * @param index The index from which to fetch the layer
   *
   * @return Layer at specified index
   */
  std::shared_ptr<Layer> getLayer(int index) const;

  /**
   * @brief This method will return the output layer (the last layer of the
   * network)
   *
   * @return The output Layer
   */
  std::shared_ptr<Layer> getOutputLayer() const;

  /**
   * @brief This method will get you the number of layers currently in the
   * Network
   *
   * @return A size_t representing the number of layers in the Network
   */
  size_t getNumLayers() const;

  /**
   * @brief Get the slug of the network based on it's architecture
   *
   * @return A string representing the combined slug of the different components
   * in the Network
   */
  std::string getSlug() const;

  /**
   * @brief This method will Train the model with the given inputs and labels
   *
   * @param X The inputs that will be passed to the model
   * @param y The labels that represent the expected outputs of the model
   * @param epochs
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   * @param progBar Ouput a progress bar for the training process . Default:
   * `true`
   *
   * @return The last training's loss
   */
  double train(std::vector<std::vector<double>> X, std::vector<double> y,
               int epochs = 1,
               const std::vector<std::shared_ptr<Callback>> callbacks = {},
               bool progBar = true);

  /**
   * @brief This method will Train the model with the given inputs and labels
   *
   * @param inputs The inputs that will be passed to the model
   * @param labels The labels that represent the expected outputs of the model
   * @param epochs
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   * @param progBar Whether to output a progress bar for the training process.
   * Default: `true`
   *
   * @return The last training's loss
   */
  double train(std::vector<std::vector<std::vector<double>>> X,
               std::vector<double> y, int epochs = 1,
               const std::vector<std::shared_ptr<Callback>> callbacks = {},
               bool progBar = true);

  /**
   * @brief This method will train the model with the given TrainingData
   *
   * @param trainingData the data passed through the TrainingData class
   * @param epochs
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   * @param progBar Whether to output a progress bar for the training
   * process. Default: `true`
   *
   * @return The last training's loss
   */
  double train(
      TrainingData<std::vector<std::vector<double>>, std::vector<double>>
          trainingData,
      int epochs = 1,
      const std::vector<std::shared_ptr<Callback>> callbacks = {},
      bool progBar = true);

  /**
   * @brief This method will train the model with the given TrainingData
   *
   * @param trainingData the data passed through the TrainingData class
   * @param epochs
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   * @param progBar Whether to output a progress bar for the training process.
   * Default: `true`
   *
   * @return The last training's loss
   */
  double train(TrainingData<std::vector<std::vector<std::vector<double>>>,
                            std::vector<double>>
                   trainingData,
               int epochs = 1,
               const std::vector<std::shared_ptr<Callback>> callbacks = {},
               bool progBar = true);

  /**
   * @brief This model will try to make predictions based off the inputs passed
   *
   * @param inputs The inputs that will be passed through the network
   *
   * @return This method will return the outputs of the neural network
   */
  Eigen::MatrixXd predict(std::vector<std::vector<double>> inputs);

  /**
   * @brief This model will try to make predictions based off the inputs passed
   *
   * @param inputs The inputs that will be passed through the network
   *
   * @return This method will return the outputs of the neural network
   */
  Eigen::MatrixXd predict(std::vector<std::vector<std::vector<double>>> inputs);

  /**
   * @brief Save the current model to a binary file
   *
   * @param filename the name of the file in which to save the model params
   */
  void to_file(const std::string &filename) override {
    // Serializing model to a binary file
    std::ofstream file(filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(file);
    archive(*this);
  }

  /**
   * @brief Load a model's params from a file
   *
   * @param filename the name of the from which to load the model params
   */
  void from_file(const std::string &filename) override {
    // Making sure the file exists and is binary
    assert(fileExistsWithExtension(filename, ".bin") &&
           "The file doesn't exists or is not binary '.bin'");

    // Deserializing the model from the binary file
    std::ifstream file(filename, std::ios::binary);
    cereal::BinaryInputArchive archive(file);
    archive(*this);
  }

  ~Network();

 private:
  // non-public serialization
  friend class cereal::access;

  std::vector<std::shared_ptr<Layer>> layers;
  LOSS lossFunc =
      LOSS::QUADRATIC;  // Storing the loss function for serialization
  bool progBar = true;
  double (*cmpLoss)(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
  Eigen::MatrixXd (*cmpLossGrad)(const Eigen::MatrixXd &,
                                 const Eigen::MatrixXd &);
  std::shared_ptr<Optimizer> optimizer;

  template <class Archive>
  void save(Archive &archive) const {
    archive(cereal::base_class<Model>(this), layers, lossFunc);
    archive.serializeDeferments();
  };

  template <class Archive>
  void load(Archive &archive) {
    archive(cereal::base_class<Model>(this), layers, lossFunc);
    setLoss(lossFunc);
  }

  /**
   * @brief online training with given training data
   *
   * @tparam D1 The type of data passed
   * @tparam D2 The type of labels passed
   * @param inputs A vector of inputs (features) of type D1
   * @param labels A vector of labels (targets) of type D2. Each element in this
   * vector corresponds to the label of the training example at the same index
   * in the inputs vector.
   * @param epochs An integer specifying the number of times the training
   * algorithm should iterate over the dataset.
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   *
   * @return A double value that represents the average loss of the training
   * process. This can be used to gauge the effectiveness of the process.
   *
   * @note The functions assumes that the inputs and labels will be of the same
   * length.
   */
  template <typename D1, typename D2>
  double onlineTraining(std::vector<D1> inputs, std::vector<D2> labels,
                        int epochs,
                        std::vector<std::shared_ptr<Callback>> callbacks = {});

  /**
   * @brief mini-batch training with given training data
   *
   * @tparam D1 The type of data passed
   * @tparam D2 The type of labels passed
   * @param trainingData A `TrainingData` object
   * @param epochs An integer specifying the number of times the training
   * algorithm should iterate over the dataset.
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   *
   * @return A double value that represents the average loss of the training
   * process. This can be used to gauge the effectiveness of the process.
   *
   * @note The functions assumes that the inputs and labels will be of the same
   * length.
   */
  template <typename D1, typename D2>
  double trainer(TrainingData<D1, D2> trainingData, int epochs,
                 std::vector<std::shared_ptr<Callback>> callbacks = {});

  /**
   * @brief mini-batch training with given training data
   *
   * @tparam D1 The type of data passed
   * @tparam D2 The type of labels passed
   * @param trainingData A `TrainingData` object
   * @param epochs An integer specifying the number of times the training
   * algorithm should iterate over the dataset.
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   * @return A double value that represents the average loss of the
   * training process. This can be used to gauge the effectiveness of the
   * process.
   *
   * @note The functions assumes that the inputs and labels will be of the same
   * length.
   */
  template <typename D1, typename D2>
  double miniBatchTraining(
      TrainingData<D1, D2> trainingData, int epochs,
      std::vector<std::shared_ptr<Callback>> callbacks = {});

  /**
   * @brief batch training with given training data
   *
   * @tparam D1 The type of data passed
   * @tparam D2 The type of labels passed
   * @param trainingData A `TrainingData` object
   * @param epochs An integer specifying the number of times the training
   * algorithm should iterate over the dataset.
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   * @return A double value that represents the average loss of the training
   * process. This can be used to gauge the effectiveness of the process.
   *
   * @note The functions assumes that the inputs and labels will be of the same
   * length.
   */
  template <typename D1, typename D2>
  double batchTraining(TrainingData<D1, D2> trainingData, int epochs,
                       std::vector<std::shared_ptr<Callback>> callbacks = {});

  /**
   * @brief This method will pass the inputs through the network and return an
   * output
   *
   * @param inputs The inputs that will be passed through the network
   *
   * @return The output of the network
   */
  Eigen::MatrixXd forwardProp(
      std::vector<std::vector<std::vector<double>>> &inputs,
      bool training = false);

  /**
   * @brief This method will pass the inputs through the network and return an
   * output
   *
   * @param inputs The inputs that will be passed through the network
   *
   * @return The output of the network
   */
  Eigen::MatrixXd forwardProp(std::vector<std::vector<double>> &inputs,
                              bool training = false);

  /**
   * @brief This method will pass the inputs through the network and return an
   * output
   *
   * @param inputs The inputs that will be passed through the network
   *
   * @return The output of the network
   */
  Eigen::MatrixXd forwardProp(Eigen::MatrixXd &inputs, bool training = false);

  Eigen::MatrixXd feedForward(Eigen::MatrixXd inputs, int startIdx = 0,
                              bool training = false);

  /**
   * @brief This method will compute the loss and backpropagate it through the
   * network whilst doing adjusting the parameters accordingly.
   *
   * @param outputs The outputs from the forward propagation
   * @param y The expected outputs (targets)
   */
  void backProp(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y);

  /**
   * @brief This method will go over the provided callbacks and trigger the
   * appropriate methods whilst passing the necessary logs.
   *
   * @param checkpointName The name of the checkpoint (e.g. onTrainBegin,
   * onEpochEnd, etc.)
   * @param callbacks A vector of `Callback` that will be called during training
   * stages
   */
  void trainingCheckpoint(std::string checkpointName,
                          std::vector<std::shared_ptr<Callback>> callbacks);

  /**
   * @brief This method will compute the accuracy of the model based on the
   * outputs of the model and the expected values.
   *
   * @param outputs The outputs from the forward propagation
   * @param y The expected outputs (targets)
   *
   * @return The accuracy of the model (percentage of correct predictions)
   */
  double computeAccuracy(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y);

  /**
   * @brief This method will update the optimizer's setup
   *
   * @param numLayers The number of layers in the network
   */
  void updateOptimizerSetup(size_t numLayers);
};
}  // namespace NeuralNet

CEREAL_REGISTER_TYPE(NeuralNet::Network);

CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNet::Model, NeuralNet::Network);