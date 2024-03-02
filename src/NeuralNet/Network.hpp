#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include <cereal/cereal.hpp> // for defer
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "Model.hpp"
#include "utils/Formatters.hpp"
#include "utils/Functions.hpp"
#include "utils/Gauge.hpp"
#include "optimizers/Optimizer.hpp"
#include "layers/Layer.hpp"
#include "layers/Flatten.hpp"
#include "layers/Dense.hpp"
#include "optimizers/optimizers.hpp"
#include "losses/losses.hpp"
#include "data/Tensor.hpp"
#include "data/TrainingData.hpp"
#include "callbacks/Callback.hpp"

namespace NeuralNet
{
  class Layer;

  class Network : public Model
  {
  public:
    Network();

    /**
     * @brief Method that sets up the model's hyperparameter
     *
     * @param optimizer An Optimizer's child class
     * @param epochs The number of epochs
     * @param loss The loss function
     */
    void setup(const std::shared_ptr<Optimizer> &optimizer, LOSS loss = LOSS::QUADRATIC);

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
     * @brief This method will return the output layer (the last layer of the network)
     *
     * @return The output Layer
     */
    std::shared_ptr<Layer> getOutputLayer() const;

    /**
     * @brief This method will get you the number of layers currently in the Network
     *
     * @return A size_t representing the number of layers in the Network
     */
    size_t getNumLayers() const;

    /**
     * @brief This method will Train the model with the given inputs and labels
     *
     * @param inputs The inputs that will be passed to the model
     * @param labels The labels that represent the expected outputs of the model
     * @param epochs
     * @param callbacks A vector of `Callback` that will be called during training stages
     *
     * @return The last training's loss
     */
    double train(std::vector<std::vector<double>> inputs, std::vector<double> labels, int epochs = 1, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief This method will Train the model with the given inputs and labels
     *
     * @param inputs The inputs that will be passed to the model
     * @param labels The labels that represent the expected outputs of the model
     * @param epochs
     * @param callbacks A vector of `Callback` that will be called during training stages
     *
     * @return The last training's loss
     */
    double train(std::vector<std::vector<std::vector<double>>> inputs, std::vector<double> labels, int epochs = 1, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief This method will train the model with the given TrainingData
     *
     * @param trainingData the data passed through the TrainingData class
     * @param epochs
     * @param callbacks A vector of `Callback` that will be called during training stages
     *
     * @return The last training's loss
     */
    double train(TrainingData<std::vector<std::vector<double>>, std::vector<double>> trainingData, int epochs = 1, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief This method will train the model with the given TrainingData
     *
     * @param trainingData the data passed through the TrainingData class
     * @param epochs
     * @param callbacks A vector of `Callback` that will be called during training stages
     *
     * @return The last training's loss
     */
    double train(TrainingData<std::vector<std::vector<std::vector<double>>>, std::vector<double>> trainingData, int epochs = 1, std::vector<std::shared_ptr<Callback>> callbacks = {});

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

    ~Network();

  private:
    // non-public serialization
    friend class cereal::access;

    template <class Archive>
    void save(Archive &archive) const
    {
      archive(layers, lossFunc);
      archive.serializeDeferments();
    };

    template <class Archive>
    void load(Archive &archive)
    {
      archive(layers, lossFunc);
      setLoss(lossFunc);
    }

    double loss = 0, accuracy = 0;
    std::vector<std::shared_ptr<Layer>> layers;
    LOSS lossFunc;      // Storing the loss function for serialization
    int cp = 0, tp = 0; // Correct Predictions, Total Predictions
    bool debugMode = false;
    double (*cmpLoss)(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
    Eigen::MatrixXd (*cmpLossGrad)(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
    std::shared_ptr<Optimizer> optimizer;

    /**
     * @brief online training with given training data
     *
     * @tparam D1 The type of data passed
     * @tparam D2 The type of labels passed
     * @param inputs A vector of inputs (features) of type D1
     * @param labels A vector of labels (targets) of type D2. Each element in this vector corresponds to the
     * label of the training example at the same index in the inputs vector.
     * @param epochs An integer specifying the number of times the training algorithm should iterate over the dataset.
     * @param callbacks A vector of `Callback` that will be called during training stages
     *
     * @return A double value that represents the average loss of the training process. This can be used to gauge the effectiveness of the process.
     *
     * @note The functions assumes that the inputs and labels will be of the same length.
     */
    template <typename D1, typename D2>
    double onlineTraining(std::vector<D1> inputs, std::vector<D2> labels, int epochs, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief mini-batch training with given training data
     *
     * @tparam D1 The type of data passed
     * @tparam D2 The type of labels passed
     * @param inputs A vector of inputs (features) of type D1
     * @param labels A vector of labels (targets) of type D2. Each element in this vector corresponds to the
     * label of the training example at the same index in the inputs vector.
     * @param epochs An integer specifying the number of times the training algorithm should iterate over the dataset.
     * @param callbacks A vector of `Callback` that will be called during training stages
     *
     * @return A double value that represents the average loss of the training process. This can be used to gauge the effectiveness of the process.
     *
     * @note The functions assumes that the inputs and labels will be of the same length.
     */
    template <typename D1, typename D2>
    double trainer(TrainingData<D1, D2> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief mini-batch training with given training data
     *
     * @tparam D1 The type of data passed
     * @tparam D2 The type of labels passed
     * @param inputs A vector of inputs (features) of type D1
     * @param labels A vector of labels (targets) of type D2. Each element in this vector corresponds to the
     * label of the training example at the same index in the inputs vector.
     * @param epochs An integer specifying the number of times the training algorithm should iterate over the dataset.
     * @param callbacks A vector of `Callback` that will be called during training stages     * @return A double value that represents the average loss of the training process. This can be used to gauge the effectiveness of the process.
     *
     * @note The functions assumes that the inputs and labels will be of the same length.
     */
    template <typename D1, typename D2>
    double miniBatchTraining(TrainingData<D1, D2> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief batch training with given training data
     *
     * @tparam D1 The type of data passed
     * @tparam D2 The type of labels passed
     * @param inputs A vector of inputs (features) of type D1
     * @param labels A vector of labels (targets) of type D2. Each element in this vector corresponds to the
     * label of the training example at the same index in the inputs vector.
     * @param epochs An integer specifying the number of times the training algorithm should iterate over the dataset.
     * @param callbacks A vector of `Callback` that will be called during training stages
     * @return A double value that represents the average loss of the training process. This can be used to gauge the effectiveness of the process.
     *
     * @note The functions assumes that the inputs and labels will be of the same length.
     */
    template <typename D1, typename D2>
    double batchTraining(TrainingData<D1, D2> trainingData, int epochs, std::vector<std::shared_ptr<Callback>> callbacks = {});

    /**
     * @brief This method will pass the inputs through the network and return an output
     *
     * @param inputs The inputs that will be passed through the network
     *
     * @return The output of the network
     */
    Eigen::MatrixXd forwardProp(std::vector<std::vector<std::vector<double>>> &inputs);

    /**
     * @brief This method will pass the inputs through the network and return an output
     *
     * @param inputs The inputs that will be passed through the network
     *
     * @return The output of the network
     */
    Eigen::MatrixXd forwardProp(std::vector<std::vector<double>> &inputs);

    /**
     * @brief This method will pass the inputs through the network and return an output
     *
     * @param inputs The inputs that will be passed through the network
     *
     * @return The output of the network
     */
    Eigen::MatrixXd forwardProp(Eigen::MatrixXd &inputs);

    Eigen::MatrixXd feedForward(Eigen::MatrixXd inputs, int startIdx = 0);

    /**
     * @brief This method will compute the loss and backpropagate it through the network whilst doing adjusting the parameters accordingly.
     *
     * @param outputs The outputs from the forward propagation
     * @param y The expected outputs (targets)
     */
    void backProp(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y);

    /**
     * @brief This method will go over the provided callbacks and trigger the appropriate methods whilst passing the necessary logs.
     *
     * @param checkpointName The name of the checkpoint (e.g. onTrainBegin, onEpochEnd, etc.)
     * @param callbacks A vector of `Callback` that will be called during training stages
     */
    void trainingCheckpoint(std::string checkpointName, std::vector<std::shared_ptr<Callback>> callbacks);

    /**
     * @brief This method will compute the accuracy of the model based on the outputs of the model and the expected values.
     *
     * @param outputs The outputs from the forward propagation
     * @param y The expected outputs (targets)
     *
     * @return The accuracy of the model (percentage of correct predictions)
     */
    double computeAccuracy(Eigen::MatrixXd &outputs, Eigen::MatrixXd &y);

    /**
     * @brief This method will fetch the logs and return them
     *
     * @return A map of useful logs
     */
    std::unordered_map<std::string, double> getLogs();

    /**
     * @brief This method will update the optimizer's setup
     *
     * @param numLayers The number of layers in the network
     */
    void updateOptimizerSetup(size_t numLayers);
  };
} // namespace NeuralNet
