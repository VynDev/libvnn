/*
* @Author: Vyn
* @Date:   2019-04-18 15:01:42
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-01 19:26:52
*/

#include <iostream>

#include "vtest/vtest.hpp"
#include "../includes/Vyn/NeuralNetwork/All.h"

TEST(NEURON)
{
	CASE("Test: setting activation function & bias")
	{
		Vyn::NeuralNetwork::Neuron neuron0(0);
		REQUIRE(neuron0.GetActivationFunctionId() == 0);
		Vyn::NeuralNetwork::Neuron neuron1(1);
		REQUIRE(neuron1.GetActivationFunctionId() == 1);
		Vyn::NeuralNetwork::Neuron neuron2(2);
		REQUIRE(neuron2.GetActivationFunctionId() == 2);

		Vyn::NeuralNetwork::Neuron neuronNotBias(NEURON_FUNCTION_SIGMOID);
		REQUIRE(!neuronNotBias.IsBias());
		Vyn::NeuralNetwork::Neuron neuronBias(NEURON_FUNCTION_BIAS);
		REQUIRE(neuronBias.IsBias());
	}

	CASE("Test: affecting value")
	{
		Vyn::NeuralNetwork::Neuron neuron(0);
		neuron.SetValue(0);
		REQUIRE(neuron.GetValue() == 0);
		neuron.SetValue(-42);
		REQUIRE(neuron.GetValue() == -42);
		neuron.SetValue(42);
		REQUIRE(neuron.GetValue() == 42);
		neuron.SetValue(0.4564);
		REQUIRE(neuron.GetValue() == 0.4564);
	}

	CASE("Test: connecting neurons & compute value & activation function")
	{
		Vyn::NeuralNetwork::Neuron inputNeuron(NEURON_FUNCTION_NONE);
		Vyn::NeuralNetwork::Neuron outputNeuron(NEURON_FUNCTION_SIGMOID);

		REQUIRE(inputNeuron.GetInputConnections().size() == 0);
		REQUIRE(inputNeuron.GetOutputConnections().size() == 0);

		REQUIRE(outputNeuron.GetInputConnections().size() == 0);
		REQUIRE(outputNeuron.GetOutputConnections().size() == 0);

		inputNeuron.ConnectTo(&outputNeuron);
		REQUIRE(inputNeuron.GetInputConnections().size() == 0);
		REQUIRE(inputNeuron.GetOutputConnections().size() == 1);

		REQUIRE(inputNeuron.GetOutputConnections()[0]->GetOutput() == &outputNeuron);
		
		REQUIRE(outputNeuron.GetInputConnections().size() == 1);
		REQUIRE(outputNeuron.GetOutputConnections().size() == 0);

		REQUIRE(outputNeuron.GetInputConnections()[0]->GetInput() == &inputNeuron);

		inputNeuron.GetOutputConnections()[0]->SetWeight(0.5);
		inputNeuron.SetValue(1);

		outputNeuron.ComputeValue();
		outputNeuron.ActivateFunction();
		REQUIRE(outputNeuron.GetRawValue() == 0.5);
		REQUIRE(outputNeuron.GetValue() == Vyn::NeuralNetwork::Sigmoid(&outputNeuron, 0.5));

		inputNeuron.GetOutputConnections()[0]->SetWeight(0);
		inputNeuron.SetValue(1);

		outputNeuron.ComputeValue();
		outputNeuron.ActivateFunction();
		REQUIRE(outputNeuron.GetRawValue() == 0);
		REQUIRE(outputNeuron.GetValue() == Vyn::NeuralNetwork::Sigmoid(&outputNeuron, 0));

		inputNeuron.GetOutputConnections()[0]->SetWeight(17);
		inputNeuron.SetValue(0.25);

		outputNeuron.ComputeValue();
		outputNeuron.ActivateFunction();
		REQUIRE(outputNeuron.GetRawValue() == 4.25);
		REQUIRE(outputNeuron.GetValue() == Vyn::NeuralNetwork::Sigmoid(&outputNeuron, 4.25));

		inputNeuron.GetOutputConnections()[0]->SetWeight(0.1);
		inputNeuron.SetValue(-1);

		outputNeuron.ComputeValue();
		outputNeuron.ActivateFunction();
		REQUIRE(outputNeuron.GetRawValue() == -0.1);
		REQUIRE(outputNeuron.GetValue() == Vyn::NeuralNetwork::Sigmoid(&outputNeuron, -0.1));
	}
	
	TEST_EXIT(TEST_SUCCESS);
}
