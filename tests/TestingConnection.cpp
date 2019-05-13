/*
* @Author: Vyn
* @Date:   2019-04-18 15:01:42
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-13 16:03:36
*/

#include <iostream>

#include "vtest/vtest.hpp"
#include "Vyn/NeuralNetwork.h"

using namespace Vyn::NeuralNetwork;

TEST(CONNECTION)
{
	CASE("Basic")
	{
		Vyn::NeuralNetwork::Neuron		inputNeuron;
		Vyn::NeuralNetwork::Neuron		outputNeuron;
		Vyn::NeuralNetwork::Connection	connection;

		REQUIRE(connection.GetInput() == nullptr);
		REQUIRE(connection.GetOutput() == nullptr);
		REQUIRE(connection.GetWeight() == 0);

		connection.SetInput(&inputNeuron);
		connection.SetOutput(&outputNeuron);

		REQUIRE(connection.GetInput() == &inputNeuron);
		REQUIRE(connection.GetOutput() == &outputNeuron);

		connection.SetWeight(42);
		REQUIRE(connection.GetWeight() == 42);
		connection.SetWeight(0.42);
		REQUIRE(connection.GetWeight() == 0.42);
		connection.SetWeight(-0.42);
		REQUIRE(connection.GetWeight() == -0.42);
	}

	CASE("Back propagation")
	{
		Vyn::NeuralNetwork::Connection	connection;

		REQUIRE(connection.GetGradient() == 0);

		connection.SetGradient(0.42);
		REQUIRE(connection.GetGradient() == 0.42);

		REQUIRE(connection.ShouldUpdate() == false);
		connection.SetShouldUpdate(true);
		REQUIRE(connection.ShouldUpdate() == true);
	}
	
	TEST_EXIT(TEST_SUCCESS);
}