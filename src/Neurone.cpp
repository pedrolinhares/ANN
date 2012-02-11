#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "Neurone.h"


void Neurone::setInput (const std::vector<double>& _input) {
	input = _input;

	if (input.size() != weights.size()) {
		weights.clear();
		for (int i = 0; i < input.size(); i++) {		
			double num = ((double) rand() / double(RAND_MAX));
			weights.push_back(num);			
		}
		bias = (double) rand() / double(RAND_MAX);
	}
}

double Neurone::output() {
	//funcao soma
	sum = 0.0;
 	for (int i = 0; i < input.size(); i++)
		sum += input[i] * weights[i];
 	sum += bias;

 	//funcao sigmoid
 	_output = 1.0 / (1.0 + exp(-sum));

 	return _output;
}

std::vector<double>  Neurone::calculateSigma(double correctOutput, bool isDownStream = false) {
 	_downStream.clear();
 	sig = _output * (1.0 - _output) * (correctOutput - _output);

	if (isDownStream) {
		for (auto weight : weights)
			_downStream.push_back(weight * sig);
		return _downStream;
	}
	_downStream.push_back(sig);
	return _downStream;
}

std::vector<double> Neurone::calculateSigma(const std::vector<double> downStream, bool isDownStream = false) {
	_downStream.clear();
	double sum_donwStream = std::accumulate(downStream.begin(), downStream.end(), 0.0);
	sig = _output * (1.0 - _output) * sum_donwStream;

	if (isDownStream) {
		for (auto weight : weights)
			_downStream.push_back(weight * sig);
		return _downStream;
	}
	_downStream.push_back(sig);
	return _downStream;
}

std::vector<double> Neurone::getSigma() {
	return _downStream;
}

void Neurone::updateWeights(double learningRate) {
	//delta rule
	for (int i = 0; i < weights.size(); i++) {
		weights[i] += learningRate * sig * input[i];
	}
	bias += learningRate * sig;
}