/*
* @Author: Vyn
* @Date:   2019-03-10 18:33:50
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 14:09:31
*/

#include "Network.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

#include <iostream>
#include <cmath>

namespace Vyn
{

	namespace NeuralNetwork
	{
		Value			PredictOne(Network *network, const Values &inputs, const Values &expectedOutputs)
		{
			network->Predict(inputs);
			return (network->GetCost(expectedOutputs));
		}

		Value			Network::TrainBatch(Network *network, std::vector<Values> &inputs, std::vector<Values> &expectedOutputs, int batchSize, int i)
		{
			Value	totalCost;
			Values	derivedCosts;

			totalCost = 0;
			const Neurons &outputLayerNeurons = network->GetOutputLayer()->GetNeurons();
			derivedCosts.reserve(outputLayerNeurons.size());

			for (Neurons::size_type k = 0; k < outputLayerNeurons.size(); ++k)
			{
				derivedCosts.push_back(0);
			}
			if (inputs.size() - i < batchSize)
				batchSize = inputs.size() - i;
			for (int j = 0; j < batchSize; ++j)
			{
				totalCost += PredictOne(network, inputs[i + j], expectedOutputs[i + j]);
				for (Neurons::size_type k = 0; k < outputLayerNeurons.size(); ++k)
				{
					derivedCosts[k] = derivedCosts[k] + network->GetDerivedCost(expectedOutputs[i + j], outputLayerNeurons[k]);
				}
			}
			totalCost = totalCost / batchSize;
			for (Values::size_type k = 0; k < outputLayerNeurons.size(); ++k)
			{
				derivedCosts[k] = derivedCosts[k] / batchSize;
			}
			if (totalCost > errorPropagationLimit)
				network->Propagate(expectedOutputs[i], derivedCosts);
			return (totalCost);
		}

		Value					ValidationSet(Network *network, TrainingParameters_t &parameters)
		{
			Value	totalCost;

			totalCost = 0;
			for (int i = 0; i < parameters.validationSetInputs.size(); ++i)
			{
				totalCost += PredictOne(network, parameters.validationSetInputs[i], parameters.validationSetOutputs[i]);
			}
			totalCost = totalCost / parameters.validationSetInputs.size();
			return (totalCost);
		}

		void					Network::Fit(TrainingParameters_t parameters, int batchSize, int nbIteration)
		{
			Value					totalCost;
			Value					validationCost = -1;
			Value					lastValidationCost = -1;
			int						noImprovementEpoch = 0;
			int						noImprovementEpochLimit = 0;
			int						k;

			//std::cout << "Training, batch size = " << batchSize << ", nbIteration = " << nbIteration << std::endl;
			if (parameters.trainingCsv)
			{
				*(parameters.trainingCsv) << parameters.trainingSetInputs.size() << std::endl;
				*(parameters.trainingCsv) << batchSize << std::endl;
			}
			if (parameters.validationCsv)
			{
				*(parameters.validationCsv) << parameters.validationSetInputs.size() << std::endl;
				*(parameters.validationCsv) << parameters.validationSetInputs.size() << std::endl;
			}
			for (int i = 0; i < nbIteration; ++i)
			{
				totalCost = 0;
				k = 0;
				for (int j = 0; j < parameters.trainingSetInputs.size(); j += batchSize)
				{
					Value		cost;
					cost = TrainBatch(this, parameters.trainingSetInputs, parameters.trainingSetOutputs, batchSize, j);
					totalCost += cost;
					++k;
					if (parameters.trainingCsv != nullptr)
						*(parameters.trainingCsv) << cost << std::endl;
					DEBUG_CHECK_VALUE(cost, "Cost");
				}
				totalCost = totalCost / k;
				std::cout << "[" << i << "] Cost: " << totalCost << "    (learning rate: " << GetLearningRate() << ")" << std::endl;
				if (parameters.validationCsv != nullptr)
				{
					validationCost = ValidationSet(this, parameters);
					*(parameters.validationCsv) << validationCost << std::endl;
				}
				// early stopping
				if (earlyStoppingEnabled && validationCost != -1)
				{
					++noImprovementEpoch;
					if (lastValidationCost == -1 || validationCost < lastValidationCost)
						noImprovementEpoch = 0;
					if (noImprovementEpoch > noImprovementEpochLimit)
					{
						std::cout << "Training finished (early stopping)" << std::endl;
						return ;
					}
					lastValidationCost = validationCost;
				}
				
			}
			//std::cout << "Training finished" << std::endl;
		}
	}
}