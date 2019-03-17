/*
* @Author: Vyn
* @Date:   2019-03-04 14:44:05
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-16 18:19:36
*/

#include <cmath>
#include <string>
#include <iostream>

#include "Network.h"
#include "types.h"
#include "Neuron.h"
#include "Connection.h"

using namespace vyn::neuralnetwork;

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

value_t	CrossEntropy(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput)
{
	value_t		total;

	total = 0;
	for (std::vector<value_t>::size_type i = 0; i < outputNeurons.size(); ++i)
	{
		DEBUG_CHECK_VALUE(outputNeurons[i]->GetValue(), "Neuron ouputs " + std::to_string(i));
		if (outputNeurons[i]->GetValue() > 0.00001 && outputNeurons[i]->GetValue() < 0.99)
			total += -(expectedOutput[i] * log(outputNeurons[i]->GetValue()) + (1 - expectedOutput[i]) * log((1 - outputNeurons[i]->GetValue())));
		else
			total += expectedOutput[i] * (1 - outputNeurons[i]->GetValue()) + (1 - expectedOutput[i]) * outputNeurons[i]->GetValue();
	}
	DEBUG_CHECK_VALUE(total, "Total error");
	return (total);
}

value_t	CrossEntropyDerivative(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput, Neuron *outputNeuron)
{
	value_t		result;
	for (std::vector<value_t>::size_type i = 0; i < outputNeurons.size(); ++i)
	{
		if (outputNeuron->GetId() == outputNeurons[i]->GetId())
		{
			DEBUG_CHECK_VALUE(expectedOutput[i], "Expected output");
			DEBUG_CHECK_VALUE(outputNeuron]->GetValue(), "Neuron output");
			if ((expectedOutput[i] == 1 && outputNeuron->GetValue() >= 0.999999) || (expectedOutput[i] == 0 && outputNeuron->GetValue() <= 0.0000001))
				return (0);
			if (expectedOutput[i] == 1 && outputNeuron->GetValue() < 0.0000001)
				return (-5);
			if (expectedOutput[i] == 0 && outputNeuron->GetValue() > 0.999999)
				return (5);
			result = -((expectedOutput[i] / outputNeuron->GetValue()) - ((1 - expectedOutput[i]) / (1 - outputNeuron->GetValue())));
			DEBUG_CHECK_VALUE(result, "Derivative of cost function result error. Expected neuron output: " + std::to_string(expectedOutput[i]) + ", neuron ouput: " + std::to_string(outputNeuron->GetValue()));
			return (result);
		}
	}
	throw std::string("Can't derivate cost function with respect to neuron");
}