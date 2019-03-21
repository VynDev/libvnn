/*
* @Author: Vyn
* @Date:   2019-03-10 18:33:50
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-17 17:55:11
*/

#include "Network.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

#include <iostream>
#include <cmath>

namespace vyn::neuralnetwork {

	static value_t			PredictOne(Network *network, std::vector<value_t> &columns, std::vector<value_t> &expectedOutputs)
	{
		std::vector<value_t>		inputs;

		for (std::vector<int>::size_type j = 0; j < columns.size(); ++j)
			inputs.push_back(columns[j]);
		network->Predict(inputs);
		return (network->GetCost(expectedOutputs));
	}

	static value_t			TrainBatch(Network *network, std::vector<std::vector<value_t>> &inputs, std::vector<std::vector<value_t>> &expectedOutputs, int batchSize, int i)
	{
		value_t								totalCost;
		std::vector<Neuron *>				outputLayerNeurons;
		std::vector<value_t>				derivedCosts;

		totalCost = 0;
		outputLayerNeurons = network->GetOutputLayer()->GetNeurons();

		for (std::vector<Neuron *>::size_type k = 0; k < outputLayerNeurons.size(); ++k)
		{
			derivedCosts.push_back(0);
		}
		if (inputs.size() - i < batchSize)
			batchSize = inputs.size() - i;
		for (int j = 0; j < batchSize; ++j)
		{
			totalCost += PredictOne(network, inputs[i + j], expectedOutputs[i + j]);
			for (std::vector<Neuron *>::size_type k = 0; k < outputLayerNeurons.size(); ++k)
			{
				derivedCosts[k] = derivedCosts[k] + network->GetDerivedCost(expectedOutputs[i + j], outputLayerNeurons[k]);
			}
		}
		for (std::vector<value_t>::size_type k = 0; k < outputLayerNeurons.size(); ++k)
		{
			derivedCosts[k] = derivedCosts[k] / batchSize;
		}
		network->Propagate(expectedOutputs[i], derivedCosts);
		return (totalCost / batchSize);
	}

	void					Network::Fit(std::vector<std::vector<value_t>> inputs, std::vector<std::vector<value_t>> outputs, int batchSize, int nbIteration, std::stringstream *csv)
	{
		std::cout << "Training, batch size = " << batchSize << ", nbIteration = " << nbIteration << std::endl;
		for (int i = 0; i < nbIteration; ++i)
		{
			//if (i % 1000 == 0 && i != 0)
				//SetLearningRate(GetLearningRate() * 0.90);
			std::cout << "Iteration n°" << i << std::endl;
			for (int j = 0; j < inputs.size(); j += batchSize)
			{
				value_t		cost;
				cost = TrainBatch(this, inputs, outputs, batchSize, j);
				if (csv != nullptr)
					*csv << cost << std::endl;
				std::cout << "Cost: " << cost << std::endl;
				DEBUG_CHECK_VALUE(cost, "Cost");
			}
		}
		std::cout << "Training finished" << std::endl;
	}
}