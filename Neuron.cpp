#include"Neuron.hpp"

Neuron::Neuron(){
	value = 0;
	sigmaAcumulate = 0;
	//activationFunc = &tanH;
	//dActivationFunc = &dTanH;
	activationFunc = &relu;
	dActivationFunc = &dRelu;
}
Neuron::~Neuron(){}
void Neuron::setValue(float _value) { value = _value; }
float Neuron::getValue() { return value; }
float Neuron::getActivatedValue() { return activationFunc(value); }
float Neuron::getDiffActivatedValue() { return dActivationFunc(value); }
void Neuron::addToValue(float _value) { value += _value; }
void Neuron::addToSigmaAcumulate(float sigma) { sigmaAcumulate += sigma; }
float Neuron::getSigmaAcumulate() { return sigmaAcumulate; }
void Neuron::clearSigmaAcumulate() { sigmaAcumulate = 0; }

void Neuron::setActivationFunc(float (*activation)(float), float (*dActivation)(float)) {
	activationFunc = activation;
	dActivationFunc = dActivation;
}
void Neuron::clearNeuron() {
	value = 0;
	sigmaAcumulate = 0;
}