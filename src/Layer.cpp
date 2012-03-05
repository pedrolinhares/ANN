#include "Layer.h"
#include <iostream>
#include <sstream>

Layer::Layer(int numOfNeurones): hidden(true) {
	for (int i = 0; i < numOfNeurones; i++)
		neurones.push_back (Neurone());
}

void Layer::setHidden(bool value) {
	hidden = value;
}

void Layer::setInput(const std::vector<double>& input) {
	int size = neurones.size();

	for (int i = 0; i < size; i++)
		neurones[i].setInput(input);
}

void Layer::calculate_output() {
	output.clear();
	for (int i = 0; i < neurones.size(); i++)
		output.push_back(neurones[i].output());
}

std::vector<double> Layer::getOutput() {
	return output;
}

std::vector<std::vector<double> > Layer::getLayerDownStream() {
	std::vector<std::vector<double> > layerDownStream;
	for (int i = 0; i < neurones.size(); i++) {
		layerDownStream.push_back(neurones[i].getSigma());
	}
	return layerDownStream;
}

void Layer::updateWeights(const std::vector<double>& correctOutput, double learningRate) {
	for (int i = 0; i < neurones.size(); i++) {
		neurones[i].calculateSigma(correctOutput[i], true);	
		neurones[i].updateWeights(learningRate);	
	}
}

void Layer::updateWeights(const std::vector<std::vector<double> >& downStream, double learningRate) {
	for (int i = 0; i < neurones.size(); i++) {
	  std::vector<double> nextLayerStream;
		
		for (int j = 0; j < downStream.size(); j++)
			nextLayerStream.push_back (downStream[j][i]);
		neurones[i].calculateSigma(nextLayerStream, true);
		neurones[i].updateWeights(learningRate);
	}
}

std::string Layer::printWeights() {
	std::stringstream out(std::stringstream::in | std::stringstream::out);
	for (int i = 0; i < neurones.size(); i++) {
		out << neurones[i].printWeights();
		out << '\n';
	}

	return out.str();
}

void Layer::updateWeightsFromFile(std::ifstream &file) {
	std::string line;

	for (int i = 0; i < neurones.size(); i++) {
		getline (file, line);
		std::vector<double> lineInput;
		std::stringstream stream (line);
		std::string value;
		
		while (getline (stream, value, ' '))
			lineInput.push_back (atof (value.c_str()));
		neurones[i].setWeights(lineInput);
	}
}