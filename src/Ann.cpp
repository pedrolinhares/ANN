#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include "Ann.h"

ANN::ANN (int numberLayers, ...):learningRate(0.5) {
	va_list arguments;

	va_start (arguments, numberLayers);

	//numero de sinais de entrada
	numElementsInInput = va_arg(arguments, int);

	for (int i = 1; i < numberLayers; i++) {
		int numOfNeuronesInLayer = va_arg(arguments, int);
		Layer layer(numOfNeuronesInLayer);
		layers.push_back(layer);
	}
	va_end (arguments);

	//ultima camada n eh oculta e sim de saida
	layers[numberLayers - 2].setHidden(false);
}

void ANN::trainAnn (const char* filename, int maxEpochs, const float desiredError) {
	//read file
	std::ifstream file;
	std::string line;
	double error = 1;
	int epochs = 1;

	file.open (filename);

	if (file.is_open())
		while (/*fabs(error) > desiredError && */epochs <= maxEpochs) {
			if (file.good()) {
				getline (file, line);
				std::vector<double> lineInput;
				std::vector<double> lineOutput;

				//get input from file
				std::stringstream stream (line);
				std::string value;
				while (getline (stream, value, ' '))
					lineInput.push_back (atof (value.c_str()));

				//get desired output
				getline(file, line);
				std::stringstream outstream (line);
				while (getline (outstream, value, ' '))
					lineOutput.push_back (atof (value.c_str()));

				//passar entrada para a rede e calcula saida
					if (lineInput.size() == numElementsInInput) {
						std::vector<double> annOutput =  execute(lineInput);

						std::cout << "Iteration: " << epochs << " ==> ";
						for (double input : lineInput)
							std::cout << input << " ";

						std::cout << "\t : ";
						for (double output : annOutput)
							std::cout << output << " ";

						std::cout << "\t Desired output: ";
						for (double desired_output : lineOutput)
							std::cout << desired_output << " ";

						std::cout << "\n";

						//calcular erro e atualizar os pesos
						std::vector<double> verror (annOutput.size(), 0.0);
						for (int i = 0; i < annOutput.size(); i++)
							verror[i] = lineOutput[i] - annOutput[i];

						double sum = 0.0;
						for (double partialError : verror)
							sum += partialError;
						error = sum / double(verror.size());

						updateNet(lineOutput);
						//std::cout << error << " " << epochs <<  std::endl;
					} else
						std::cerr << "error: Number of input elements didn't match" << std::endl;

			} else {
				if (file.eof())
					file.seekg(0, std::ios::beg);
				}
			epochs++;
		}
	else
		std::cout << "Unable to read file." << std::endl;
}

std::vector<double> ANN::execute(const std::vector<double>& input) {
	layers[0].setInput (input);
	layers[0].calculate_output();

	int size = layers.size();

	//calcular saida
	for (int i = 1; i < size; i++) {
		layers[i].setInput (layers[i-1].getOutput());
		layers[i].calculate_output();
	}

	return layers[size - 1].getOutput();
}

void ANN::updateNet(const std::vector<double> correctOutput) {
	int size = layers.size();

	//calcula o erro da camada de saida
	layers[size - 1].updateWeights(correctOutput, learningRate);

	//calcula o erro das camadas escondidas propagando o erro das
	//camadas a frente
	for (int i = size - 2; i >= 0; i--) {
		layers[i].updateWeights(layers[i + 1].getLayerDownStream(), learningRate);
	}
}