#include "MLP.hpp"
#include<time.h>
#include<stdlib.h>
#include<iostream>

MLP::MLP(int _inputSize, int _howManyHiddenLayers, int* _hiddenLayerSize, int _outputSize) {
	hiddenLayerSize = _hiddenLayerSize;
	if (_howManyHiddenLayers >= 1) howManyHiddenLayers = _howManyHiddenLayers;
	else howManyHiddenLayers = 1;
	inputSize = _inputSize;
	outputSize = _outputSize;
	learningRate = 0.1;
	//errorFunc = &binaryError;
	//dErrorFucn = &dBinaryError;
	errorFunc = &simpleError;
	dErrorFucn = &dSimpleError;
	howManyConnectionsLayers = howManyHiddenLayers + 1;
	srand(time(NULL));
	bias = float(rand() % 1000) / 1000;
	includeBias = true;

	hiddenSize = 0;
	connectionLayerSize = (int*)calloc(howManyConnectionsLayers, sizeof(int));
	connectionLayerSize[0] = inputSize * hiddenLayerSize[0];
	for (int i = 1; i < howManyConnectionsLayers - 1; i++) connectionLayerSize[i] = hiddenLayerSize[i - 1] * hiddenLayerSize[i];
	connectionLayerSize[howManyConnectionsLayers - 1] = hiddenLayerSize[howManyHiddenLayers - 1] * outputSize;

	//allLayersSize = (int*)malloc(sizeof(howManyHiddenLayers + 2)); <-- FOUND!!!
	allLayersSize = (int*)malloc(sizeof(int)*(howManyHiddenLayers + 2));
	allLayersSize[0] = inputSize;
	allLayersSize[howManyHiddenLayers + 1] = outputSize;
	for (int i = 0; i < howManyHiddenLayers; i++) allLayersSize[i + 1] = hiddenLayerSize[i];

	//---CREATE NEURONS---

	//inputLayer = new Neuron[inputSize];
	inputLayer = (Neuron*)malloc(sizeof(Neuron) * inputSize);
	for(int i = 0; i < inputSize; i++) new (&inputLayer[i]) Neuron();
	//outputLayer = new Neuron[outputSize];
	outputLayer = (Neuron*)malloc(sizeof(Neuron) * outputSize);
	for (int i = 0; i < outputSize; i++) new (&outputLayer[i]) Neuron();
	for (int i = 0; i < howManyHiddenLayers; i++) hiddenSize += hiddenLayerSize[i];
	//hiddenLayers = new Neuron[hiddenSize];
	hiddenLayers = (Neuron*)malloc(sizeof(Neuron) * hiddenSize);
	for (int i = 0; i < hiddenSize; i++) new (&hiddenLayers[i]) Neuron();

	//---CREATE CONNECTIONS---

	howManyNeurons = hiddenSize + inputSize + outputSize;
	howManyConnections = hiddenLayerSize[0]*inputSize;
	for (int i = 1; i < howManyHiddenLayers; i++) howManyConnections += hiddenLayerSize[i - 1] * hiddenLayerSize[i];
	howManyConnections += hiddenLayerSize[howManyHiddenLayers - 1] * outputSize;
	connections = (Connection*)malloc(sizeof(Connection) * howManyConnections);

	for (int i = 0; i < inputSize; i++)
		for (int j = 0; j < hiddenLayerSize[0]; j++)
			new (&connections[i * hiddenLayerSize[0] + j]) Connection(&inputLayer[i], &hiddenLayers[j]);

	int tempConnectionCounter = inputSize * hiddenLayerSize[0];
	int tempNeuronInHiddenLayer[2] = { 0, hiddenLayerSize[0] };
	for (int layer = 1; layer < howManyHiddenLayers; layer++) {
		for (int i = 0; i < hiddenLayerSize[layer - 1]; i++)
			for (int j = 0; j < hiddenLayerSize[layer]; j++)
				new (&connections[tempConnectionCounter + i * hiddenLayerSize[layer] + j]) Connection(&hiddenLayers[tempNeuronInHiddenLayer[0]+i], &hiddenLayers[tempNeuronInHiddenLayer[1]+j]);
		tempConnectionCounter += hiddenLayerSize[layer - 1] * hiddenLayerSize[layer];
		tempNeuronInHiddenLayer[0] += hiddenLayerSize[layer - 1];
		tempNeuronInHiddenLayer[1] += hiddenLayerSize[layer];
	}

	for (int i = 0; i < hiddenLayerSize[howManyHiddenLayers-1]; i++)
		for (int j = 0; j < outputSize; j++)
			new (&connections[tempConnectionCounter + i * outputSize + j]) Connection(&hiddenLayers[tempNeuronInHiddenLayer[0]+i], &outputLayer[j]);

	tempConnectionCounter = 0;
	firstConnectionInLayer = (Connection**)malloc(sizeof(Connection*) * howManyConnections);
	for (int i = 0; i < howManyConnectionsLayers; i++) {
		firstConnectionInLayer[i] = &(connections[tempConnectionCounter]);
		tempConnectionCounter += connectionLayerSize[i];
	}

	//---BIAS CONNECTIONS---

	howManyBiasConnections = hiddenSize + outputSize;
	biasOne.setValue(1);
	biasOne.setActivationFunc(linear, dLinear);
	biasConnections = (Connection*)malloc(sizeof(Connection) * (howManyBiasConnections));
	for (int i = 0; i < hiddenSize; i++) new (&biasConnections[i]) Connection(&biasOne, &hiddenLayers[i]);
	for (int i = 0; i < outputSize; i++) new (&biasConnections[hiddenSize + i]) Connection(&biasOne, &outputLayer[i]);
	
	tempConnectionCounter = 0;
	firstBiasConnectionInLayer = (Connection**)malloc(sizeof(Connection*) * howManyConnectionsLayers);
	for (int i = 0; i < howManyConnectionsLayers-1; i++) {
		firstBiasConnectionInLayer[i] = &(biasConnections[tempConnectionCounter]);
		tempConnectionCounter += hiddenLayerSize[i];
	}
	firstBiasConnectionInLayer[howManyConnectionsLayers-1] = &(biasConnections[tempConnectionCounter]);

	biasConnectionLayerSize = (int*)malloc(sizeof(int) * howManyConnectionsLayers);
	for (int i = 0; i < howManyConnectionsLayers; i++) biasConnectionLayerSize[i] = allLayersSize[i + 1];
}

MLP::~MLP() {
	std::cout << "DELETING MLP\n";
}

void MLP::clearMemory() {
	std::cout << "CLEARING MEMORY\n";
	free(connectionLayerSize);
	//delete[] inputLayer;
	free(inputLayer);
	free(outputLayer);
	free(hiddenLayers);
	free(connections);
	free(biasConnections);
}

int MLP::getHowManyConnections() { return howManyConnections; }
int MLP::getHowManyNeurons() { return howManyNeurons; }
void MLP::getOutput(float* output) { for (int i = 0; i < outputSize; i++) output[i] = outputLayer[i].getActivatedValue(); }
void MLP::setLearningRate(float _learningRate) { learningRate = _learningRate; }
void MLP::setIncludeBias(bool _includeBias) { includeBias = _includeBias; }

void MLP::setActivationFuncForLayer(int layer, float (*activation)(float), float (*diffActivation)(float)) {
	if (layer == 0) for (int i = 0; i < inputSize; i++) inputLayer[i].setActivationFunc(activation, diffActivation);
	else if (layer > 0 && layer <= howManyHiddenLayers) {
		int tempCount = 0;
		for (int i = 0; i < layer - 2; i++) tempCount += hiddenLayerSize[i];
		for (int i = 0; i < hiddenLayerSize[layer - 1]; i++) hiddenLayers[i].setActivationFunc(activation, diffActivation);
	}
	else for (int i = 0; i < outputSize; i++) outputLayer[i].setActivationFunc(activation, diffActivation);
}

void MLP::printScheme() {
	std::cout << "MLP SCHEME\n";
	std::cout << inputSize << "N -> x" << connectionLayerSize[0] << "C ->";
	for(int i = 0; i < howManyHiddenLayers; i++)
		std::cout << hiddenLayerSize[i] << "N -> x" << connectionLayerSize[i+1] << "C ->";
	std::cout << outputSize << "N\n";
}

void MLP::resetNeuronValues() {
	for (int i = 0; i < inputSize; i++) inputLayer[i].clearNeuron();
	for (int i = 0; i < outputSize; i++) outputLayer[i].clearNeuron();
	for (int i = 0; i < hiddenSize; i++) hiddenLayers[i].clearNeuron();
	biasOne.clearSigmaAcumulate();
}


bool MLP::pushForward(float* _inputTab) {
	resetNeuronValues();
	for (int i = 0; i < inputSize; i++) inputLayer[i].setValue(_inputTab[i]);

	int tempConnectionCounter = 0;
	int tempBiasConnectionCounter = 0;
	int biasConnectionSize = 0;
	for (int layer = 0; layer < howManyConnectionsLayers; layer++) {

		if (layer < howManyHiddenLayers) biasConnectionSize = hiddenLayerSize[layer];
		else biasConnectionSize = outputSize;

		for (int i = 0; i < connectionLayerSize[layer]; i++) connections[tempConnectionCounter + i].executePush();
		if(includeBias) for (int i = 0; i < biasConnectionSize; i++) biasConnections[tempBiasConnectionCounter + i].executePush();
		tempConnectionCounter += connectionLayerSize[layer];
		tempBiasConnectionCounter += biasConnectionSize;
	}
	return 1; 
}

bool MLP::bakcPropagation(float* _inputTab, float* values) {
	pushForward(_inputTab);
	float* outputTab = (float*)malloc(sizeof(float) * outputSize);
	getOutput(outputTab);
	for (int i = 0; i < outputSize; i++)
		outputLayer[i].addToSigmaAcumulate(dErrorFucn(outputTab[i], values[i]));
	free(outputTab);
	if (!includeBias) {
		for (int i = howManyConnections - 1; i >= 0; i--)
			connections[i].calcSigmaAndActualizeWeights(learningRate);
	}
	else {
		Connection* tempConnectnionPtr = &connections[howManyConnections - 1];
		Connection* tempBiasConnectionPtr = &biasConnections[howManyBiasConnections - 1];
		int biasSize = outputSize;
		for (int layer = howManyConnectionsLayers-1; layer >= 0; layer--) {
			for (int i = 0; i < connectionLayerSize[i]; i++) {
				tempConnectnionPtr->calcSigmaAndActualizeWeights(learningRate);
				tempConnectnionPtr--;
			}
			for (int i = 0; i < biasSize; i++) {
				tempBiasConnectionPtr->calcSigmaAndActualizeWeights(learningRate);
				tempBiasConnectionPtr--;
			}
			if(layer-1 >= 0)biasSize = hiddenLayerSize[layer-1];
		}
	}
	/*else {
		for (int layer = howManyConnectionsLayers - 1; layer >= 0; layer--) {
			for (int i = 0; i < connectionLayerSize[layer]; i++) firstConnectionInLayer[layer][i].calcSigmaAndActualizeWeights(learningRate);
			for (int i = 0; i < biasConnectionLayerSize[layer]; i++) firstBiasConnectionInLayer[layer][i].calcSigmaAndActualizeWeights(learningRate);
		}
	}*/

	return 1;
}