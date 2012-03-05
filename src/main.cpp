#include <iostream>
#include <sstream>
#include <cstdlib>
#include "Ann.h"

int main (void) {
 	ANN ann (3, 7, 5, 3);

	//treinar
	int max_epochs = 50000;
	const float desired_error = (const float) 0.001;
	ann.trainAnn("data_sets/pos_maos.data", max_epochs, desired_error);
	ann.saveToFile("rede.net");

	//carregar rede neural a partir do arquivo salvo apos treinamento
	//ann.readFromFile("rede.net");

	//rodar exemplos

	std::string input = "";
	std::cout << " \n Input: " << std::endl;
	getline(std::cin, input);
	while (input.compare("exit") != 0) {
		std::string value;
		std::vector<double> realInput;
		std::stringstream inputStream (input);
		while (getline (inputStream, value, ' '))
			realInput.push_back (atof (value.c_str()));

		std::vector<double> output = ann.execute(realInput);

		for (double in : realInput)
			std::cout << in << " ";

		std::cout << "  saida: ";
		for (double out : output)
			std::cout << out << " ";

		std::cout << std::endl;

		std::cout << " \n Input: " << std::endl;
		getline(std::cin, input);
	}
}
