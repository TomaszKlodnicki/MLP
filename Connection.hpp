#pragma once
#include "Neuron.hpp"

#ifndef CONNECTION_CLASS
#define CONNECTION_CLASS

class Connection {
private:
	float weight;
	Neuron* start;
	Neuron* end;
	float sigma;
public:
	Connection(Neuron* _start, Neuron* _end);
	Connection(Neuron* _start, Neuron* _end, float _weight);
	~Connection();
	void executePush();
	void actualizeWeight(float learningRate);
	float getSigma();
	void calcSigma();
	void calcSigma(float differential);
	void calcSigmaAndActualizeWeights(float learningRate);
};

#endif // !CONNECTION_CLASS
