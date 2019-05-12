/*
* @Author: Vyn
* @Date:   2019-02-02 11:29:39
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-12 16:02:37
*/

#include <iostream>
#include <sstream>
#include <cmath>
#include <fstream>
#include <float.h>

#include "Network.h"
#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

#define ABS(x) (x < 0 ? -x : x)

namespace Vyn
{
	namespace NeuralNetwork
	{
		Network::Network()
		{

		}

		void					Network::AddLayer(int nbNeuron, int neuronType, int weightInitializationFunctionId)
		{
			Layer	*newLayer;
			newLayer = new Layer(nbNeuron, neuronType, weightInitializationFunctionId);
			if (layers.size() != 0)
			{
				layers[layers.size() - 1]->AddBias();
			}
			AddLayer(newLayer);
		}

		void					Network::AddLayer(Layer *layer)
		{
			Layer	*lastLayer;

			layer->SetParentNetwork(this);
			if (layers.size() == 0)
			{
				layers.push_back(layer);
				return ;
			}
			lastLayer = layers[layers.size() - 1];
			lastLayer->ConnectTo(layer);
			layers.push_back(layer);
		}

		void					Network::SetCostFunction(int id)
		{
			costFunctionId = id;
			if (id == COST_FUNCTION_MSE)
			{
				costFunction = &SquaredError;
				costFunctionDerivative = &SquaredErrorDerivative;
			}
			else if (id == COST_FUNCTION_CE)
			{
				costFunction = &CrossEntropy;
				costFunctionDerivative = &CrossEntropyDerivative;
			}
		}

		const Values	&Network::Predict(Values const &inputs)
		{
			const Neurons &outputLayerNeurons = layers[layers.size() - 1]->GetNeurons();
			this->lastOutputValues.clear();
			this->lastOutputValues.reserve(outputLayerNeurons.size());

			if (layers.size() < 2)
				throw std::string("There must be minimum 2 layers (input & output)");
			const Neurons &inputLayerNeurons = layers[0]->GetNeurons();
			if (inputLayerNeurons.size() - layers[0]->GetBiasCount() != inputs.size())
				throw std::string("Missing inputs");
			for (Neurons::size_type i = 0; i < inputs.size(); ++i)
				inputLayerNeurons[i]->SetValue(inputs[i]);
			for (Layers::size_type i = 1; i < layers.size(); ++i)
				layers[i]->ComputeValues();
			for (Neurons::size_type i = 0; i < outputLayerNeurons.size(); ++i)
				lastOutputValues.push_back(outputLayerNeurons[i]->GetValue());
			return (lastOutputValues);
		}

		Value					Network::GetCost(Values const &expectedOutput)
		{
			if (costFunction != nullptr)
				return ((*costFunction)(this->lastOutputValues, expectedOutput));
			throw std::string("No cost function defined");
		}

		Value					Network::GetDerivedCost(Values const &expectedOutput, int neuronIndex)
		{
			if (costFunctionDerivative != nullptr)
				return ((*costFunctionDerivative)(this->lastOutputValues, expectedOutput, neuronIndex));
			throw std::string("No cost function derivative defined");
		}

		Layer					*Network::GetInputLayer() const
		{
			if (layers.size() > 0)
				return (layers[0]);
			return (nullptr);
		}

		Layer					*Network::GetOutputLayer() const
		{
			if (layers.size() > 1)
				return (layers[layers.size() - 1]);
			return (nullptr);
		}

		void					Network::SaveTo(std::string fileName)
		{
			std::ofstream file;
			file.open(fileName);
			file << VYN_NEURALNETWORK_STRING << std::endl;
			file << VYN_NEURALNETWORK_VERSION << std::endl;
			file << layers.size() << std::endl;
			for (Layers::size_type i = 0; i < layers.size(); ++i)
			{
				Neurons neurons;
				neurons = layers[i]->GetNeurons();
				for (Neurons::size_type j = 0; j < neurons.size(); ++j)
				{
					file << neurons[j]->GetActivationFunctionId() << " ";
				}
				file << std::endl;
			}
			file << this->GetCostFunctionId() << std::endl;

			for (Connections::size_type i = 0; i < connections.size(); ++i)
			{
				file << connections[i]->GetWeight() << " ";
			}
			file.close();
		}

		void					Network::RandomizeConnectionsWeight()
		{
			for (int i = 0; i < connections.size(); ++i)
			{
				connections[i]->SetWeight(connections[i]->GetInput()->GetParentLayer()->NewWeightValue());
			}
		}

		void					Network::SaveTrainingCsv(std::string path)
		{
			std::ofstream file;
			file.open(path);
			file << trainingCsv.str();
			file.close();
		}

		void					Network::SaveValidationCsv(std::string path)
		{
			std::ofstream file;
			file.open(path);
			file << validationCsv.str();
			file.close();
		}
	}
}