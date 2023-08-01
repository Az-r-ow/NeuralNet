#pragma once

#include <vector>
#include <cstdlib>
#include <memory>
#include "Layer.hpp"

using std::vector;

class Network
{
public:
    Network(double learnRate = 0.1);
    void addLayer(Layer &layer);
    Layer getLayer(int index);
    Layer &getOutputLayer();
    int getNumLayer() const;
    void train(vector<vector<double>> inputs, vector<double> labels);
    void predict(vector<double> outputs);
    ~Network();

private:
    vector<Layer> layers;
    double learningRate;
    int neuronPerLayer;
    double loss = 1;

    /* Private Methods */
    Layer &getOutputLayer();
    void forwardProp(vector<double> inputs);
    void backProp(Labels y);

    /* private static functions */
    static double computeLoss(MatrixXd &o, Label &y);
    static MatrixXd computeLossDer(MatrixXd &o, Labels &y);
    static MatrixXd computeSigmoidDer(MatrixXd &a);
};
