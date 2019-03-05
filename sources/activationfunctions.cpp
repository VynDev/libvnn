/*
* @Author: Vyn
* @Date:   2019-03-03 14:49:12
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-04 11:08:13
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