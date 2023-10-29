#include "ActivationFunc.hpp"

float sigmoid(float input) {
	return 1 / (1 + exp(-input));
}
float dSigmoid(float input) {
	return exp(-input) / pow(exp(-input) + 1, 2);
}

float tanH(float input) {
	return (exp(input) - exp(-input)) / (exp(input) + exp(-input));
}
float dTanH(float input) {
	return 4 / pow(exp(-input) + exp(input), 2);
}

float relu(float input) {
	if (input > 0) return input;
	return 0;
}
float dRelu(float input) {
	if (input > 0) return 1;
	return 0;
}

float linear(float input) {
	return input;
}
float dLinear(float input) {
	return 1;
}

void softmax(float* input, int inputSize) {
	float sum = 0;
	for (int i = 0; i < inputSize; i++) {
		input[i] = exp(input[i]);
		sum += input[i];
	}
	for (int i = 0; i < inputSize; i++) input[i] /= sum;
}

float simpleCost(float* predictions, float* values, int size) {
	float sol = 0;
	for (int i = 0; i < size; i++) sol += pow(predictions[i] - values[i], 2);
	sol /= size;
	return sol;
}

float meanAbsoluteError(float* predictions, float* values, int size) {
	float sol = 0;
	for (int i = 0; i < size; i++) sol += fabs(predictions[i] - values[i]);
	sol /= size;
	return sol;
}
float meanSquasedError(float* predictions, float* values, int size) {
	float sol = 0;
	for (int i = 0; i < size; i++) sol += pow(fabs(predictions[i] - values[i]), 2);
	sol /= 2 * size;
	return sol;
}
float binaryCostFunction(float* predictions, float* values, int size) {
	float sol = 0;
	for (int i = 0; i < size; i++) sol -= values[i] * log(predictions[i]) + (1 - values[i]) * log(1 - predictions[i]);
	sol /= size;
	return sol;
}
float multiClassClassificationCostFunction(float* predicions, float* values, int size, int inputOutputSize) {
	float sol = 0;
	for (int i = 0; i < inputOutputSize; i++)
		for (int j = 0; j < size; j++)
			sol += values[i * size + j];
	sol /= size;
	return sol;
}
float simpleError(float prediction, float value) {
	return pow(prediction - value, 2);
}
float dSimpleError(float prediction, float value) {
	return (prediction - value) / 2;
}
