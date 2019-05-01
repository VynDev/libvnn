#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#define COST_FUNCTION_MSE 1
#define COST_FUNCTION_CE 2

#include <vector>
#include "Types.h"

Vyn::NeuralNetwork::value_t	SquaredError(std::vector<Vyn::NeuralNetwork::Neuron *> outputNeurons, std::vector<Vyn::NeuralNetwork::value_t> expectedOutput);
Vyn::NeuralNetwork::value_t	SquaredErrorDerivative(std::vector<Vyn::NeuralNetwork::Neuron *> outputNeurons, std::vector<Vyn::NeuralNetwork::value_t> expectedOutput, Vyn::NeuralNetwork::Neuron *outputNeuron);

Vyn::NeuralNetwork::value_t	CrossEntropy(std::vector<Vyn::NeuralNetwork::Neuron *> outputNeurons, std::vector<Vyn::NeuralNetwork::value_t> expectedOutput);
Vyn::NeuralNetwork::value_t	CrossEntropyDerivative(std::vector<Vyn::NeuralNetwork::Neuron *> outputNeurons, std::vector<Vyn::NeuralNetwork::value_t> expectedOutput, Vyn::NeuralNetwork::Neuron *outputNeuron);

#endif