#pragma once
#include"ActivationFunc.hpp"

#ifndef NEURON_CLASS
#define NEURON_CLASS

class Neuron {
private:
	float value;
	float sigmaAcumulate;
	float (*activationFunc)(float);
	float (*dActivationFunc)(float);
public:
	Neuron();
	~Neuron();
	float getValue();
	float getActivatedValue();
	float getDiffActivatedValue();
	void setValue(float _value);
	void addToValue(float _value);
	float getSigmaAcumulate();
	void clearNeuron();
	void addToSigmaAcumulate(float sigma);
	void setActivationFunc(float (*activation)(float), float (*dActivation)(float));
};

#endif // !NEURON_CLASS
