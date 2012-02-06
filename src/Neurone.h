#ifndef NEURONE_H
#define NEURONE_H

#include <vector>

class Neurone {
	public:
		Neurone ():sum(0), _output(0), bias(0.8){};
		void setInput (const std::vector<double>& _input);
		double output ();
		std::vector<double> calculateSigma(double correctOutput, bool isDownStream);
		std::vector<double> calculateSigma(const std::vector<double> downStream, bool isDownStream);
		std::vector<double> getSigma();
		void updateWeights(double learningRate);
	private:
		double sum;
		double _output;
		double bias;
		double sig;
		std::vector<double> weights;
		std::vector<double> input;
		std::vector<double> _downStream;

};

#endif