/*
* @Author: Vyn
* @Date:   2019-03-04 14:44:05
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-04 14:49:54
*/

#include <cmath>
#include <string>

#include "types.h"
#include "Neuron.h"

value_t	SquaredError(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput)
{
	value_t		total;

	total = 0;
	for (std::vector<value_t>::size_type i = 0; i < outputNeurons.size(); ++i)
	{
		total += std::pow(outputNeurons[i]->GetValue() - expectedOutput[i], 2) / 2;
	}
	return (total);
}

value_t	SquaredErrorDerivative(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput, Neuron *outputNeuron)
{
	for (std::vector<value_t>::size_type i = 0; i < outputNeurons.size(); ++i)
	{
		if (outputNeuron->GetId() == outputNeurons[i]->GetId())
			return (outputNeurons[i]->GetValue() - expectedOutput[i]);
	}
	throw std::string("Can't derivate cost function with respect to neuron");
}