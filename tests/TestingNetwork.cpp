/*
* @Author: Vyn
* @Date:   2019-04-18 15:01:42
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 15:23:45
*/

#include <iostream>

#include "vtest/vtest.hpp"
#include "../includes/Vyn/NeuralNetwork/All.h"

using Vyn::NeuralNetwork::Value;

TEST(NETWORK)
{
	CASE("Training XOR")
	{
		Vyn::NeuralNetwork::Network network;

		network.AddLayer(2, NEURON_FUNCTION_NONE, WEIGHT_INIT_0, 1);
		network.AddLayer(2, NEURON_FUNCTION_SIGMOID, WEIGHT_INIT_0, 1);
		network.AddLayer(1, NEURON_FUNCTION_SIGMOID, WEIGHT_INIT_0, 0);
		network.SetLearningRate(0.1);
		network.SetCostFunction(COST_FUNCTION_MSE);

		Vyn::NeuralNetwork::TrainingParameters	parameters;

		std::vector<std::vector<Value>>			inputs;
		std::vector<std::vector<Value>>			outputs;

		std::vector<Value> empty;
		inputs.push_back(empty);
		outputs.push_back(empty);
		inputs.push_back(empty);
		outputs.push_back(empty);
		inputs.push_back(empty);
		outputs.push_back(empty);
		inputs.push_back(empty);
		outputs.push_back(empty);

		inputs[0].push_back(0);
		inputs[0].push_back(0);
		outputs[0].push_back(0);

		inputs[1].push_back(0);
		inputs[1].push_back(1);
		outputs[1].push_back(1);

		inputs[2].push_back(1);
		inputs[2].push_back(0);
		outputs[2].push_back(1);

		inputs[3].push_back(1);
		inputs[3].push_back(1);
		outputs[3].push_back(0);

		parameters.trainingSetInputs = inputs;
		parameters.trainingSetOutputs = outputs;

		network.Fit(parameters, 1, 20000);

		network.Predict(inputs[0]);
		//std::cout << network.GetOutputLayer()->GetNeurons()[0]->GetValue() << std::endl;
		WARN(network.GetOutputLayer()->GetNeurons()[0]->GetValue() < 0.1);
		network.Predict(inputs[1]);
		//std::cout << network.GetOutputLayer()->GetNeurons()[0]->GetValue() << std::endl;
		WARN(network.GetOutputLayer()->GetNeurons()[0]->GetValue() > 0.9);
		network.Predict(inputs[2]);
		//std::cout << network.GetOutputLayer()->GetNeurons()[0]->GetValue() << std::endl;
		WARN(network.GetOutputLayer()->GetNeurons()[0]->GetValue() > 0.9);
		network.Predict(inputs[3]);
		//std::cout << network.GetOutputLayer()->GetNeurons()[0]->GetValue() << std::endl;
		WARN(network.GetOutputLayer()->GetNeurons()[0]->GetValue() < 0.1);
	}
	
	TEST_EXIT(TEST_SUCCESS);
}