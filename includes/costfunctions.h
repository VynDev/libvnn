#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#define COST_FUNCTION_MSE 0
#define COST_FUNCTION_CE 1

#include <vector>
#include "types.h"

class Neuron;

value_t	SquaredError(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput);
value_t	SquaredErrorDerivative(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput, Neuron *outputNeuron);

value_t	CrossEntropy(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput);
value_t	CrossEntropyDerivative(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput, Neuron *outputNeuron);

#endif