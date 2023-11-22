#pragma once
#include <iostream>
#include "Connection.hpp"

#ifndef MLP_CLASS
#define MLP_CLASS

class MLP {
private:
	int* hiddenLayerSize;
	int* allLayersSize;
	int howManyHiddenLayers;
	int inputSize;
	int outputSize;
	int hiddenSize;
	float bias;
	bool includeBias;
	int howManyConnections;
	int howManyNeurons;
	int howManyConnectionsLayers;
	int howManyBiasConnections;
	int* connectionLayerSize;
	int* biasConnectionLayerSize;
	float learningRate;
	Neuron biasOne;
	Neuron* inputLayer;
	Neuron* outputLayer;
	Neuron* hiddenLayers;
	Connection* connections;
	Connection* biasConnections;
	Connection** firstConnectionInLayer;
	Connection** firstBiasConnectionInLayer;
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
	void setIncludeBias(bool _includeBias);
	void getOutput(float* output);
	void setLearningRate(float _learningRate);
	void setActivationFuncForLayer(int layer, float (*activation)(float), float (*diffActivation)(float));
};

#endif // !MPL_CLASS
