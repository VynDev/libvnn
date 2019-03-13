/*
* @Author: Vyn
* @Date:   2019-03-13 12:53:26
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-13 13:03:59
*/

#include "utils.h"

namespace vyn::neuralnetwork::scale {

	static value_t	GetMin(std::vector<std::vector<value_t>> &inputs, int i)
	{
		value_t	min;

		min = inputs[0][i];
		for (int j = 0; j < inputs.size(); ++j)
		{
			if (inputs[j][i] < min)
				min = inputs[j][i];
		}
		return (min);
	}

	static value_t	GetMax(std::vector<std::vector<value_t>> &inputs, int i)
	{
		value_t	max;

		max = inputs[0][i];
		for (int j = 0; j < inputs.size(); ++j)
		{
			if (inputs[j][i] > max)
				max = inputs[j][i];
		}
		return (max);
	}

	void	MinMax(std::vector<std::vector<value_t>> &inputs)
	{
		std::vector<value_t>	firstLine;

		firstLine = inputs[0];
		for (int i = 0; i < firstLine.size(); ++i)
		{
			value_t	min = GetMin(inputs, i);
			value_t max = GetMax(inputs, i);
			for (int j = 0; j < inputs.size(); ++j)
			{
				inputs[j][i] = (inputs[j][i] - min) / (max - min);
			}
		}
	}
}