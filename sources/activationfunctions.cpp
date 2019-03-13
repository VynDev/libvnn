/*
* @Author: Vyn
* @Date:   2019-03-03 14:49:12
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-09 11:43:06
*/

#include <cmath>

#include "types.h"
#include "Neuron.h"

value_t		Sigmoid(Neuron *neuron, value_t x)
{
	return (1 / (1 + (exp(-x))));
}

value_t		SigmoidDerivative(Neuron *neuron, value_t x)
{
	return (Sigmoid(neuron, x) * (1 - Sigmoid(neuron, x)));
}

value_t		Softmax(Neuron *neuron, value_t x)
{
	std::vector<Neuron *>	outputNeurons;
	value_t					sum;

	outputNeurons = neuron->GetParentLayer()->GetNeurons();
	for (std::vector<Neuron *>::size_type i = 0; i < outputNeurons.size(); ++i)
	{
		sum += exp(outputNeurons[i]->GetValue());
	}
	return ((exp(x) / sum));
}

value_t		SoftmaxDerivative(Neuron *neuron, value_t x)
{
	return (Softmax(neuron, 1 - Softmax(neuron, x)));
}