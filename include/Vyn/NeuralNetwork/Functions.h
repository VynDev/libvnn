#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define CROSSOVER_FUNCTION_DEFAULT 0

#include <vector>
#include "Types.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		namespace Activation
		{
			enum {None, Bias, Sigmoid, Softmax};
		}

		namespace Cost
		{
			enum {None, MSE, BCE};
		}
	
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