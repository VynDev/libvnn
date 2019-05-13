/*
* @Author: Vyn
* @Date:   2019-03-10 18:33:50
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-13 14:00:07
*/

#include "Network.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace Vyn
{

	namespace NeuralNetwork
	{
		Value			PredictOne(Network *network, const Values &inputs, const Values &expectedOutputs)
		{
			network->Predict(inputs);
			return (network->GetCost(expectedOutputs));
		}

		Value			Network::TrainBatch(Network *network, std::vector<Values> &inputs, std::vector<Values> &expectedOutputs, int batchSize, int i, std::vector<int> &indexes)
		{
			Value	totalCost;

			totalCost = 0;
			const Neurons &outputLayerNeurons = network->GetOutputLayer()->GetNeurons();
			tmpDerivedCost.clear();
			tmpDerivedCost.reserve(outputLayerNeurons.size());

			for (Neurons::size_type k = 0; k < outputLayerNeurons.size(); ++k)
			{
				tmpDerivedCost.push_back(0);
			}
			if (inputs.size() - i < batchSize)
				batchSize = inputs.size() - i;
			for (int j = 0; j < batchSize; ++j)
			{
				totalCost += PredictOne(network, inputs[indexes[i] + j], expectedOutputs[indexes[i] + j]);
				for (Neurons::size_type k = 0; k < outputLayerNeurons.size(); ++k)
				{
					tmpDerivedCost[k] = tmpDerivedCost[k] + network->GetDerivedCost(expectedOutputs[indexes[i] + j], k);
				}
			}
			totalCost = totalCost / batchSize;
			for (Values::size_type k = 0; k < outputLayerNeurons.size(); ++k)
			{
				tmpDerivedCost[k] = tmpDerivedCost[k] / batchSize;
			}
			if (totalCost > errorPropagationLimit)
				network->Propagate(expectedOutputs[i], tmpDerivedCost);
			return (totalCost);
		}

		Value					ValidationSet(Network *network, const TrainingParameters &parameters)
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

		void					Network::Fit(TrainingParameters parameters, int batchSize, int nbIteration)
		{
			Value					totalCost;
			Value					validationCost = -1;
			Value					lastValidationCost = -1;
			int						noImprovementEpoch = 0;
			int						noImprovementEpochLimit = 0;
			int						k;
			std::vector<int>		indexes;

			for (int i = 0; i < parameters.trainingSetInputs.size(); ++i)
				indexes.push_back(i);

			trainingCsv << parameters.trainingSetInputs.size() << std::endl;
			trainingCsv << batchSize << std::endl;

			validationCsv << parameters.validationSetInputs.size() << std::endl;
			validationCsv << parameters.validationSetInputs.size() << std::endl;

			for (int i = 0; i < nbIteration; ++i)
			{
				if (shuffleEnabled)
					std::random_shuffle(indexes.begin(), indexes.end());
				totalCost = 0;
				k = 0;
				for (int j = 0; j < parameters.trainingSetInputs.size(); j += batchSize)
				{
					Value		cost;
					cost = TrainBatch(this, parameters.trainingSetInputs, parameters.trainingSetOutputs, batchSize, j, indexes);
					totalCost += cost;
					++k;
					if (true)
						trainingCsv << cost << std::endl;
					DEBUG_CHECK_VALUE(cost, "Cost");
				}
				totalCost = totalCost / k;
				//std::cout << "[" << i << "] Cost: " << totalCost << "    (learning rate: " << GetLearningRate() << ")" << std::endl;

				validationCost = ValidationSet(this, parameters);
				validationCsv << validationCost << std::endl;
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