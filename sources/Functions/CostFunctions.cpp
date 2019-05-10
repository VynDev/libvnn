/*
* @Author: Vyn
* @Date:   2019-03-04 14:44:05
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 13:08:24
*/

#include <cmath>
#include <string>
#include <iostream>
#include <float.h>

#include "Network.h"
#include "Types.h"
#include "Neuron.h"
#include "Connection.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		Value	SquaredError(const Neurons &outputNeurons, const Values &expectedOutput)
		{
			Value		total;

			total = 0;
			for (Values::size_type i = 0; i < outputNeurons.size(); ++i)
			{
				total += std::pow(outputNeurons[i]->GetValue() - expectedOutput[i], 2) / 2;
			}
			return (total);
		}

		Value	SquaredErrorDerivative(const Neurons &outputNeurons, const Values &expectedOutput, Neuron *outputNeuron)
		{
			for (Values::size_type i = 0; i < outputNeurons.size(); ++i)
			{
				if (outputNeuron->GetId() == outputNeurons[i]->GetId())
					return (outputNeurons[i]->GetValue() - expectedOutput[i]);
			}
			throw std::string("Can't derivate cost function with respect to neuron");
		}

		Value	CrossEntropy(const Neurons &outputNeurons, const Values &expectedOutput)
		{
			Value		total;

			total = 0;
			for (Values::size_type i = 0; i < outputNeurons.size(); ++i)
			{
				DEBUG_CHECK_VALUE(outputNeurons[i]->GetValue(), "Neuron ouputs");

				if (expectedOutput[i] == 1)
				{
					if (outputNeurons[i]->GetValue() != 0)
						total += -log(outputNeurons[i]->GetValue());
					else
						total += -log(DBL_MIN);
				}
				else if (expectedOutput[i] == 0)
				{
					if (outputNeurons[i]->GetValue() != 1)
						total += -log(1 - outputNeurons[i]->GetValue());
					else
						total += -log(DBL_MIN);
				}
			}
			DEBUG_CHECK_VALUE(total, "Total error");
			return (total);
		}

		Value	CrossEntropyDerivative(const Neurons &outputNeurons, const Values &expectedOutput, Neuron *outputNeuron)
		{
			Value		result;
			Value		leftResult;
			Value		rightResult;

			for (Values::size_type i = 0; i < outputNeurons.size(); ++i)
			{
				if (outputNeuron->GetId() == outputNeurons[i]->GetId())
				{
					DEBUG_CHECK_VALUE(expectedOutput[i], "Expected output");
					DEBUG_CHECK_VALUE(outputNeuron->GetValue(), "Neuron output");
					if (expectedOutput[i] == 1)
					{
						result = -(1 / outputNeuron->GetValue() - 1);
						//if (result < -10)
							//return (-10);
						return (result);
					}
					else if (expectedOutput[i] == 0)
					{
						result = -(-1 / (1 - outputNeuron->GetValue()) + 1);
						//if (result > 10)
							//return (10);
						return (result);
					}
				}
			}
			throw std::string("Can't derivate cost function with respect to neuron");
		}
	}
}