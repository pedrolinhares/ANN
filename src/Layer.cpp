#include "Layer.h"
#include <iostream>

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
	for (auto neurone : neurones) {
		layerDownStream.push_back(neurone.getSigma());
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