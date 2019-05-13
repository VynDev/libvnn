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

TEST(LAYER)
{
	CASE("Add neuron & bias")
	{
		Vyn::NeuralNetwork::Layer	layer;

		REQUIRE(layer.GetNeurons().size() == 0);
		REQUIRE(layer.GetBiasCount() == 0);
		layer.AddNeuron(Activation::Sigmoid);
		REQUIRE(layer.GetNeurons().size() == 1);
		REQUIRE(layer.GetBiasCount() == 0);
		layer.AddNeuron(Activation::Sigmoid);
		REQUIRE(layer.GetNeurons().size() == 2);
		REQUIRE(layer.GetBiasCount() == 0);
		layer.AddBias();
		REQUIRE(layer.GetNeurons().size() == 3);
		REQUIRE(layer.GetBiasCount() == 1);
	}

	CASE("Basic with bias (testing connections)")
	{
		Vyn::NeuralNetwork::Layer	inputLayer(1, Activation::None, WEIGHT_INIT_0);
		Vyn::NeuralNetwork::Layer	outputLayer(1, Activation::Sigmoid, WEIGHT_INIT_0);

		REQUIRE(inputLayer.GetNeurons().size() == 1);
		REQUIRE(outputLayer.GetNeurons().size() == 1);
		REQUIRE(inputLayer.GetBiasCount() == 0);
		REQUIRE(outputLayer.GetBiasCount() == 0);

		inputLayer.ConnectTo(&outputLayer);
		std::vector<Vyn::NeuralNetwork::Neuron *>	inputNeurons;
		std::vector<Vyn::NeuralNetwork::Neuron *>	outputNeurons;
		
		inputNeurons = inputLayer.GetNeurons();
		outputNeurons = outputLayer.GetNeurons();
		for (int i = 0; i < inputNeurons.size(); ++i)
		{
			for (int j = 0; j < outputNeurons.size(); ++j)
			{
				REQUIRE(inputNeurons[i]->GetOutputConnections()[j]->GetOutput() == outputNeurons[j]);
			}
		}
	}

	CASE("Less basic (testing connections)")
	{
		Vyn::NeuralNetwork::Layer	inputLayer(42, Activation::None, WEIGHT_INIT_0);
		Vyn::NeuralNetwork::Layer	outputLayer(21, Activation::Sigmoid, WEIGHT_INIT_0);

		REQUIRE(inputLayer.GetNeurons().size() == 42);
		REQUIRE(outputLayer.GetNeurons().size() == 21);
		REQUIRE(inputLayer.GetBiasCount() == 0);
		REQUIRE(outputLayer.GetBiasCount() == 0);

		inputLayer.ConnectTo(&outputLayer);
		std::vector<Vyn::NeuralNetwork::Neuron *>	inputNeurons;
		std::vector<Vyn::NeuralNetwork::Neuron *>	outputNeurons;
		
		inputNeurons = inputLayer.GetNeurons();
		outputNeurons = outputLayer.GetNeurons();
		for (int i = 0; i < inputNeurons.size(); ++i)
		{
			for (int j = 0; j < outputNeurons.size(); ++j)
			{
				REQUIRE(inputNeurons[i]->GetOutputConnections()[j]->GetOutput() == outputNeurons[j]);
			}
		}
	}

	CASE("Compute values & get values")
	{
		Vyn::NeuralNetwork::Layer	inputLayer(2, Activation::None, WEIGHT_INIT_0);
		Vyn::NeuralNetwork::Layer	outputLayer(2, Activation::Sigmoid, WEIGHT_INIT_0);

		inputLayer.AddBias();
		inputLayer.ConnectTo(&outputLayer);
		std::vector<Vyn::NeuralNetwork::Neuron *>	inputNeurons;
		std::vector<Vyn::NeuralNetwork::Neuron *>	outputNeurons;
		
		inputNeurons = inputLayer.GetNeurons();
		outputNeurons = outputLayer.GetNeurons();
		for (int i = 0; i < inputNeurons.size(); ++i)
		{
			inputNeurons[i]->SetValue(0.5);
			for (int j = 0; j < inputNeurons[i]->GetOutputConnections().size(); ++j)
			{
				inputNeurons[i]->GetOutputConnections()[j]->SetWeight(0.5);
			}
		}
		outputLayer.ComputeValues();
		for (int i = 0; i < outputNeurons.size(); ++i)
		{
			REQUIRE(outputNeurons[i]->GetRawValue() == 0.5 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5);
			REQUIRE(outputNeurons[i]->GetValue() == Vyn::NeuralNetwork::Sigmoid(outputNeurons[i], outputNeurons[i]->GetRawValue()));
		}
		Vyn::NeuralNetwork::Values	outputValues = outputLayer.GetValues();
		for (int i = 0; i < outputNeurons.size(); ++i)
		{
			REQUIRE(outputValues[i] == (1 / (1 + exp(-(0.5 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5))))); // <- Sigmoid
		}
	}
	
	TEST_EXIT(TEST_SUCCESS);
}