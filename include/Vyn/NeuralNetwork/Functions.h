#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define NEURON_FUNCTION_NONE 0
#define NEURON_FUNCTION_BIAS 1
#define NEURON_FUNCTION_SIGMOID 2
#define NEURON_FUNCTION_SOFTMAX 3

#define COST_FUNCTION_MSE 1
#define COST_FUNCTION_CE 2

#define CROSSOVER_FUNCTION_DEFAULT 0

#include <vector>
#include "Types.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		Value Sigmoid(Neuron *, Value x);
		Value SigmoidDerivative(Neuron *, Value x);

		Value Softmax(Neuron *, Value x);
		Value SoftmaxDerivative(Neuron *, Value x);

		Value weightInitialization0(Layer *layer);
		Value weightInitialization1(Layer *layer);

		void DefaultCrossOverFunction(Population *population);

		
		Value SquaredError(const Values &outputs, const Values &expectedOutputs);
		Value SquaredErrorDerivative(const Values &outputs, const Values &expectedOutputs, int neuronIndex);

		Value CrossEntropy(const Values &outputs, const Values &expectedOutputs);
		Value CrossEntropyDerivative(const Values &outputs, const Values &expectedOutputs, int neuronIndex);

		void DefaultCrossOverFunction(Population *population);

	}
}
#endif