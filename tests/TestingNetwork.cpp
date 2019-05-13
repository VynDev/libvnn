/*
* @Author: Vyn
* @Date:   2019-04-18 15:01:42
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-13 18:31:17
*/

#include <iostream>
#include <vector>

#include "vtest/vtest.hpp"
#include "Vyn/NeuralNetwork.h"

namespace NNet = Vyn::NeuralNetwork;

TEST(NETWORK)
{
	CASE("Training XOR method 1")
	{
		srand(time(NULL));
		NNet::Network network;

		network.AddLayer(2, NNet::Activation::None);
		network.AddLayer(2, NNet::Activation::Sigmoid);
		network.AddLayer(1, NNet::Activation::Sigmoid);
		network.SetLearningRate(0.1);
		network.SetCostFunction(NNet::Cost::MSE);

		std::vector<NNet::Values> inputs =	{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
		std::vector<NNet::Values> outputs =	{{0}, 	 {1},	 {1},	 {0}};

		for (int n = 0; n < 20000; ++n) // 20000 iterations
		{
			for (int i = 0; i < inputs.size(); ++i) // Each iteration, train each input
			{
				network.Predict(inputs[i]);
				network.Propagate(outputs[i]);
			}
		}

		WARN(network.Predict(inputs[0])[0] < 0.1);
		WARN(network.Predict(inputs[1])[0] > 0.9);
		WARN(network.Predict(inputs[2])[0] > 0.9);
		WARN(network.Predict(inputs[3])[0] < 0.1);
	}

	CASE("Training XOR method 2")
	{
		srand(time(NULL));
		NNet::Network network;

		network.AddLayer(2, NNet::Activation::None);
		network.AddLayer(2, NNet::Activation::Sigmoid);
		network.AddLayer(1, NNet::Activation::Sigmoid);
		network.SetLearningRate(0.1);
		network.SetCostFunction(NNet::Cost::MSE);

		std::vector<NNet::Values> inputs =	{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
		std::vector<NNet::Values> outputs = {{0}, 	 {1},	 {1},	 {0}};

		NNet::TrainingParameters parameters;

		parameters.trainingSetInputs = inputs;
		parameters.trainingSetOutputs = outputs;

		network.Fit(parameters, 1, 20000);

		WARN(network.Predict(inputs[0])[0] < 0.1);
		WARN(network.Predict(inputs[1])[0] > 0.9);
		WARN(network.Predict(inputs[2])[0] > 0.9);
		WARN(network.Predict(inputs[3])[0] < 0.1);
	}
	
	TEST_EXIT(TEST_SUCCESS);
}