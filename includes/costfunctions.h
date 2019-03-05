#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include <vector>
#include "types.h"

class Neuron;

value_t	SquaredError(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput);
value_t	SquaredErrorDerivative(std::vector<Neuron *> outputNeurons, std::vector<value_t> expectedOutput, Neuron *outputNeuron);

#endif