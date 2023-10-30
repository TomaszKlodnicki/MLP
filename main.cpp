#include<iostream>
#include<fstream>
#include<string>
#include "MLP.hpp"

using namespace std;

struct csvData {
	float** data;
	bool* clas;
	int col;
	int rows;
};

csvData tabFromCSV(string filename, int howManyCols) {
	string line;
	ifstream MyReadFile(filename);
	float* tempPtr;
	csvData output;
	output.col = howManyCols;
	output.rows = 0;
	output.data = nullptr;
	output.clas = nullptr;
	while (getline(MyReadFile, line)) {
		tempPtr = (float*)malloc(sizeof(float) * output.col);
		int tempCounter = 0;
		string temp = "";
		for (int i = 0; i < line.length(); i++) {
			if (line[i] != ',') temp += line[i];
			else {
				tempPtr[tempCounter] = stof(temp);
				tempCounter++;
				temp = "";
			}
		}
		output.rows++;
		output.clas = (bool*)realloc(output.clas, sizeof(bool) * output.rows);
		output.clas[output.rows - 1] = bool(stof(temp));
		output.data = (float**)realloc(output.data, sizeof(float*) * output.rows);
		output.data[output.rows - 1] = tempPtr;
		tempCounter = 0;
	}
	MyReadFile.close();
	return output;
}

int main() {

	csvData trein = tabFromCSV("data/banknote_trein.txt", 4);
	cout << trein.data[0][0] << "," << trein.data[0][1] << "," << trein.data[0][2] << "," << trein.data[0][3] << "\n";

	int sizeTab[3] = { 5, 4, 5 };
	MLP neuralNetwork = MLP(5, 3, sizeTab, 2);

	cout << "Created " << neuralNetwork.getHowManyNeurons() << " neurons and " << neuralNetwork.getHowManyConnections() << " connections\n";
	neuralNetwork.printScheme();

	float input[5] = { 1, 0, 0.5, 0.2, 1 };
	float output[2] = { 0.5, 0.5 };
	float predictions[2];
	neuralNetwork.pushForward(input);
	neuralNetwork.getOutput(predictions);
	cout << predictions[0] << ", " << predictions[1] << endl;

	for (int i = 0; i < 600; i++)
		neuralNetwork.bakcPropagation(input, output);

	neuralNetwork.pushForward(input);
	neuralNetwork.getOutput(predictions);
	cout << predictions[0] << ", " << predictions[1] << endl;



	int testSize[4] = { 5, 10, 15, 20 };
	MLP test = MLP(4, 4, testSize, 1);

	test.setActivationFuncForLayer(-1, sigmoid, dSigmoid);
	//test.setActivationFuncForLayer(0, linear, dLinear);
	test.setLearningRate(0.05);

	int trueTable[2][2] = { { 0,0 },{ 0,0 } };


	for (int i = 0; i < trein.rows; i++) {
		test.pushForward(trein.data[i]);
		test.getOutput(predictions);
		//cout << predictions[0] << " -> " << trein.clas[i] << endl;
		if (predictions[0] >= 0.5) trueTable[1][int(trein.clas[i])]++;
		else trueTable[0][int(trein.clas[i])]++;
	}

	cout << "predict 0 | " << trueTable[0][0] << " |" << trueTable[0][1] << "\n";
	cout << "predict 1 | " << trueTable[1][0] << " |" << trueTable[1][1] << "\n";
	cout << "Acuracy = " << float(trueTable[0][0] + trueTable[1][1]) / float(trueTable[0][0] + trueTable[0][1] + trueTable[1][0] + trueTable[1][1]) << "\n";
	trueTable[0][0] = 0;
	trueTable[1][0] = 0;
	trueTable[0][1] = 0;
	trueTable[1][1] = 0;



	for (int j = 0; j < 50; j++) {
		for (int i = 0; i < trein.rows; i++) {
			float out = float(trein.clas[i]);
			test.bakcPropagation(trein.data[i], &out);
		}

		for (int i = 0; i < trein.rows; i++) {
			test.pushForward(trein.data[i]);
			test.getOutput(predictions);
			//cout << predictions[0] << " -> " << trein.clas[i] << endl;
			if (predictions[0] >= 0.5) trueTable[1][int(trein.clas[i])]++;
			else trueTable[0][int(trein.clas[i])]++;
		}

		cout << "predict 0 | " << trueTable[0][0] << " |" << trueTable[0][1] << "\n";
		cout << "predict 1 | " << trueTable[1][0] << " |" << trueTable[1][1] << "\n";
		cout << "Acuracy = " << float(trueTable[0][0] + trueTable[1][1]) / float(trueTable[0][0] + trueTable[0][1] + trueTable[1][0] + trueTable[1][1]) << "\n";
		trueTable[0][0] = 0;
		trueTable[1][0] = 0;
		trueTable[0][1] = 0;
		trueTable[1][1] = 0;
	}

	return 1;
}