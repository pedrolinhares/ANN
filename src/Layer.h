#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neurone.h"

class Layer {
	public:
		Layer(int numOfNeurones);	
		int size() {return neurones.size();}
		void setInput(const std::vector<double>& input);
		void calculate_output();
		void setHidden(bool value);
		std::vector<std::vector<double> > getLayerDownStream();
		void updateWeights(const std::vector<double>& correctOutput, double learningRate);
		void updateWeights(const std::vector<std::vector<double> >& downStream, double learningRate);
		std::vector<double> getOutput();

	private:
		std::vector<Neurone> neurones;
		std::vector<double> output;
		bool hidden;
};

#endif