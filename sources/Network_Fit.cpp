/*
* @Author: Vyn
* @Date:   2019-03-10 18:33:50
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-25 11:00:57
*/

#include "Network.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

#include <iostream>
#include <cmath>

namespace vyn::neuralnetwork {

	value_t			PredictOne(Network *network, std::vector<value_t> &inputs, std::vector<value_t> &expectedOutputs)
	{
		network->Predict(inputs);
		return (network->GetCost(expectedOutputs));
	}

	value_t			Network::TrainBatch(Network *network, std::vector<std::vector<value_t>> &inputs, std::vector<std::vector<value_t>> &expectedOutputs, int batchSize, int i)
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
		totalCost = totalCost / batchSize;
		for (std::vector<value_t>::size_type k = 0; k < outputLayerNeurons.size(); ++k)
		{
			derivedCosts[k] = derivedCosts[k] / batchSize;
		}
		if (totalCost > errorPropagationLimit)
			network->Propagate(expectedOutputs[i], derivedCosts);
		return (totalCost);
	}

	void					Network::Fit(std::vector<std::vector<value_t>> inputs, std::vector<std::vector<value_t>> outputs, int batchSize, int nbIteration, std::stringstream *csv)
	{
		value_t					lastTotalCost = -1;
		int						noImprovementEpoch = 0;
		int						noImprovementEpochLimit = 100;
		int						k;

		value_t					totalCost;
		std::cout << "Training, batch size = " << batchSize << ", nbIteration = " << nbIteration << std::endl;
		for (int i = 0; i < nbIteration; ++i)
		{
			totalCost = 0;
			k = 0;
			for (int j = 0; j < inputs.size(); j += batchSize)
			{
				value_t		cost;
				cost = TrainBatch(this, inputs, outputs, batchSize, j);
				totalCost += cost;
				++k;
				if (csv != nullptr)
					*csv << cost << std::endl;
				//std::cout << "Cost: " << cost << std::endl;
				DEBUG_CHECK_VALUE(cost, "Cost");
			}
			// early stopping
			totalCost = totalCost / k;
			std::cout << "[" << i << "] Cost: " << totalCost << "    (learning rate: " << GetLearningRate() << ")" << std::endl;
			++noImprovementEpoch;
			
			if (lastTotalCost == -1 || totalCost != lastTotalCost)
				noImprovementEpoch = 0;
			if (noImprovementEpoch > noImprovementEpochLimit)
			{
				std::cout << "Training finished (early stopping)" << std::endl;
				return ;
			}
			lastTotalCost = totalCost;
		}
		std::cout << "Training finished" << std::endl;
	}
}