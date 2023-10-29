#pragma once
#include <iostream>
#include "Connection.hpp"

#ifndef MLP_CLASS
#define MLP_CLASS

class MLP {
private:
	int* hiddenLayerSize;
	int howManyHiddenLayers;
	int inputSize;
	int outputSize;
	int hiddenSize;
	float bias;
	int howManyConnections;
	int howManyNeurons;
	int howManyConnectionsLayers;
	int* connectionLayerSize;
	float learningRate;
	Neuron* inputLayer;
	Neuron* outputLayer;
	Neuron* hiddenLayers;
	Connection* connections;
	void resetNeuronValues();
	float (*errorFunc)(float, float);
	float (*dErrorFucn)(float, float);
public:
	MLP(int _inputSize, int _howManyHiddenLayers, int* _hiddenLayerSize, int _outputSize);
	~MLP();
	bool pushForward(float* _inputTab);
	bool bakcPropagation(float* _inputTab, float* values);
	int getHowManyConnections();
	int getHowManyNeurons();
	void printScheme();
	void getOutput(float* output);
	void setLearningRate(float _learningRate);
	void setActivationFuncForLayer(int layer, float (*activation)(float), float (*diffActivation)(float));
};

#endif // !MPL_CLASS
