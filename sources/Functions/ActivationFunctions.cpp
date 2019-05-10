/*
* @Author: Vyn
* @Date:   2019-03-03 14:49:12
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 12:57:53
*/

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "Network.h"
#include "Neuron.h"
#include "Population.h"
#include "Types.h"
#include "Utils.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		Value		Sigmoid(Neuron *neuron, Value x)
		{
			Value result;

			result = 1 / (1 + (exp(-x)));
			DEBUG_CHECK_VALUE(-x, "Sigmoid x");
			DEBUG_CHECK_VALUE(result, "Sigmoid");
			return (result);
		}

		Value		SigmoidDerivative(Neuron *neuron, Value x)
		{
			Value result;

			const Value sigmoidValue = Sigmoid(neuron, x);
			result = sigmoidValue * (1 - sigmoidValue);
			DEBUG_CHECK_VALUE(x, "Sigmoid derivative x");
			DEBUG_CHECK_VALUE(result, "Sigmoid derivative");
			return (result);
		}

		Value		Softmax(Neuron *neuron, Value x)
		{
			Neurons outputNeurons;
			Value sum;
			Value result;

			sum = DBL_MIN;
			//sum = 0;
			outputNeurons = neuron->GetParentLayer()->GetNeurons();
			for (Neurons::size_type i = 0; i < outputNeurons.size(); ++i)
			{
				sum += exp(outputNeurons[i]->GetRawValue());
				DEBUG_CHECK_VALUE(sum, "Softmax sum");
			}
			result = exp(x) / sum;
			DEBUG_CHECK_VALUE(x, "Softmax x");
			DEBUG_CHECK_VALUE(exp(x), "Softmax exp(x)");
			DEBUG_CHECK_VALUE(result, "Softmax");
			return (result);
		}

		Value		SoftmaxDerivative(Neuron *neuron, Value x)
		{
			Value result;

			result = Softmax(neuron, 1 - Softmax(neuron, x));
			DEBUG_CHECK_VALUE(x, "Softmax derivative x");
			DEBUG_CHECK_VALUE(result, "Softmax derivative");
			//std::cout << "softmax derivative: " << result << std::endl;
			return (result);
		}

		Value		weightInitialization0(Layer *layer)
		{
			return (((Value)rand() / (Value)RAND_MAX) * 2 - 1);
		}

		Value		weightInitialization1(Layer *layer)
		{
			return (((Value)rand() / (Value)RAND_MAX < 0.5 ? (Value)rand() / (Value)RAND_MAX * -0.6 - 0.2 : (Value)rand() / (Value)RAND_MAX * 0.6 + 0.2));
		}
	}
}