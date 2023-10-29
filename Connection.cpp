#include "Connection.hpp"
#include <stdlib.h>
#include <time.h>

Connection::Connection(Neuron* _start, Neuron* _end) {
	start = _start;
	end = _end;
	weight = float(rand() % 1000)/1000;
	sigma = 0;
}

Connection::Connection(Neuron* _start, Neuron* _end, float _weight) {
	start = _start;
	end = _end;
	weight = _weight;
	sigma = 0;
}

Connection::~Connection() {}
float Connection::getSigma() { return sigma; }
void Connection::actualizeWeight(float learningRate) { weight -= learningRate * sigma; }
void Connection::executePush() { end->addToValue(weight * start->getActivatedValue()); }

void Connection::calcSigma(float differential) {
	sigma = differential * start->getActivatedValue() * end->getDiffActivatedValue();
	start->addToSigmaAcumulate(sigma);
}
void Connection::calcSigma() {
	calcSigma(end->getSigmaAcumulate());
};
void Connection::calcSigmaAndActualizeWeights(float learningRate) {
	calcSigma();
	actualizeWeight(learningRate);
}