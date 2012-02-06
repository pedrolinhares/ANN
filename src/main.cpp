#include <iostream>
#include <sstream>
#include <cstdlib>
#include "Ann.h"

int main (void) {
 	ANN ann (4, 6, 4, 3, 2);

	//treinar
	int max_epochs = 3000000;
	const float desired_error = (const float) 0.001;
	ann.trainAnn("data_sets/maos.txt", max_epochs, desired_error);

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