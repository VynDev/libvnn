/*
* @Author: Vyn
* @Date:   2019-03-04 14:44:05
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 14:32:31
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
		Value	SquaredError(const Values &outputs, const Values &expectedOutputs)
		{
			Value		total;

			total = 0;
			for (Values::size_type i = 0; i < outputs.size(); ++i)
			{
				total += std::pow(outputs[i] - expectedOutputs[i], 2) / 2;
			}
			return (total);
		}

		Value	SquaredErrorDerivative(const Values &outputs, const Values &expectedOutputs, int neuronIndex)
		{
			return (outputs[neuronIndex] - expectedOutputs[neuronIndex]);
			throw std::string("Can't derivate cost function with respect to neuron");
		}

		Value	CrossEntropy(const Values &outputs, const Values &expectedOutputs)
		{
			Value		total;

			total = 0;
			for (Values::size_type i = 0; i < outputs.size(); ++i)
			{
				DEBUG_CHECK_VALUE(outputs[i], "Neuron ouputs");

				if (expectedOutputs[i] == 1)
				{
					if (outputs[i] != 0)
						total += -log(outputs[i]);
					else
						total += -log(DBL_MIN);
				}
				else if (expectedOutputs[i] == 0)
				{
					if (outputs[i] != 1)
						total += -log(1 - outputs[i]);
					else
						total += -log(DBL_MIN);
				}
			}
			DEBUG_CHECK_VALUE(total, "Total error");
			return (total);
		}

		Value	CrossEntropyDerivative(const Values &outputs, const Values &expectedOutputs, int neuronIndex)
		{
			Value		result;

			if (expectedOutputs[neuronIndex] == 1)
			{
				result = -(1 / outputs[neuronIndex] - 1);
				//if (result < -10)
					//return (-10);
				return (result);
			}
			else if (expectedOutputs[neuronIndex] == 0)
			{
				result = -(-1 / (1 - outputs[neuronIndex]) + 1);
				//if (result > 10)
					//return (10);
				return (result);
			}
			throw std::string("Can't derivate cost function with respect to neuron");
		}
	}
}