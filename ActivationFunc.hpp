#pragma once
#include<math.h>

#ifndef ACTIVATION_FUNC
#define ACTIVATION_FUNC

float sigmoid(float input);
float dSigmoid(float input);
float tanH(float input);
float dTanH(float input);
float relu(float input);
float dRelu(float input);
float linear(float input);
float dLinear(float input);
void softmax(float* input, int inputSize);
#endif

#ifndef COST_FUNC
#define COST_FUNC

float simpleCost(float* predictions, float* values, int size);
float meanAbsoluteError(float* predictions, float* values, int size);
float meanSquasedError(float* predictions, float* values, int size);
float binaryCostFunction(float* predictions, float* values, int size);
float multiClassClassificationCostFunction(float* predicions, float* values, int size, int inputOutputSize);
#endif // !COST_FUNC

#ifndef ERROR_FUNC
#define ERROR_FUNC

float simpleError(float prediction, float value);
float dSimpleError(float prediction, float value);
float binaryError(float prediction, float value);
float dBinaryError(float predicion, float value);

#endif // !ERROR_FUNC


