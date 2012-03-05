#ifndef ANN_H
#define ANN_H

#include <cstdarg>
#include <vector>
#include <string>

#include "Neurone.h"
#include "Layer.h"

class ANN {
	public:
		ANN (int numberLayers, ...);
		void trainAnn(const char* filename, int maxEpochs, const float desiredError);
		std::vector<double> execute(const std::vector<double>& input);
		void updateNet(const std::vector<double> correctOutput);
		void saveToFile(const char* filename);
		void readFromFile(const char* filename); 
	
	private:	
		std::vector<Layer> layers;
		int numElementsInInput;
		double learningRate;
};

#endif