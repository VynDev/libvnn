/*
* @Author: Vyn
* @Date:   2019-03-03 14:49:12
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-16 18:37:32
*/

#include <cmath>
#include <iostream>

#include "Network.h"
#include "Neuron.h"
#include "types.h"
#include "utils.h"

namespace vyn::neuralnetwork {

	value_t		Sigmoid(Neuron *neuron, value_t x)
	{
		value_t	result;

		result = 1 / (1 + (exp(-x)));
		DEBUG_CHECK_VALUE(-x, "Sigmoid x (" + std::to_string(neuron->GetId()) + ")");
		DEBUG_CHECK_VALUE(result, "Sigmoid (" + std::to_string(neuron->GetId()) + ")");
		return (result);
	}

	value_t		SigmoidDerivative(Neuron *neuron, value_t x)
	{
		value_t	result;

		result = Sigmoid(neuron, x) * (1 - Sigmoid(neuron, x));
		DEBUG_CHECK_VALUE(x, "Sigmoid derivative x (" + std::to_string(neuron->GetId()) + ")");
		DEBUG_CHECK_VALUE(result, "Sigmoid derivative (" + std::to_string(neuron->GetId()) + ")");
		return (result);
	}

	value_t		Softmax(Neuron *neuron, value_t x)
	{
		std::vector<Neuron *>	outputNeurons;
		value_t					sum;
		value_t					result;

		sum = 0.0000001;
		outputNeurons = neuron->GetParentLayer()->GetNeurons();
		for (std::vector<Neuron *>::size_type i = 0; i < outputNeurons.size(); ++i)
		{
			sum += exp(outputNeurons[i]->GetRawValue());
		}
		result = exp(x) / sum;
		DEBUG_CHECK_VALUE(x, "Softmax x(" + std::to_string(neuron->GetId()) + ")");
		DEBUG_CHECK_VALUE(sum, "Softmax sum (" + std::to_string(neuron->GetId()) + ")");
		DEBUG_CHECK_VALUE(result, "Softmax (" + std::to_string(neuron->GetId()) + ")");
		return (result);
	}

	value_t		SoftmaxDerivative(Neuron *neuron, value_t x)
	{
		value_t	result;

		result = Softmax(neuron, 1 - Softmax(neuron, x));
		DEBUG_CHECK_VALUE(x, "Softmax derivative x (" + std::to_string(neuron->GetId()) + ")");
		DEBUG_CHECK_VALUE(result, "Softmax derivative (" + std::to_string(neuron->GetId()) + ")");
		return (result);
	}
}