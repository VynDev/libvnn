/*
* @Author: Vyn
* @Date:   2019-02-02 11:29:39
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-20 17:20:16
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

namespace vyn::neuralnetwork {

	Network::Network()
	{

	}

	void					Network::AddLayer(int nbNeuron, int neuronType, int nbBias)
	{
		Layer	*newLayer;
		newLayer = new Layer(nbNeuron, neuronType, nbBias);
		AddLayer(newLayer);
	}

	void					Network::AddLayer(Layer *layer)
	{
		Layer	*lastLayer;

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

	std::vector<value_t>	Network::Predict(std::vector<value_t> inputs)
	{
		std::vector<Neuron *>	inputLayerNeurons;
		std::vector<Neuron *>	outputLayerNeurons;
		std::vector<value_t>	outputValues;

		if (layers.size() < 2)
			throw std::string("There must be minimum 2 layers (input & output)");
		inputLayerNeurons = layers[0]->GetNeurons();
		if (inputLayerNeurons.size() - layers[0]->GetBiasCount() != inputs.size())
			throw std::string("Missing inputs");
		for (std::vector<Neuron *>::size_type i = 0; i < inputs.size(); ++i)
			inputLayerNeurons[i]->SetValue(inputs[i]);
		for (std::vector<Layer *>::size_type i = 1; i < layers.size(); ++i)
			layers[i]->ComputeValues();
		outputLayerNeurons = layers[layers.size() - 1]->GetNeurons();
		for (std::vector<Neuron *>::size_type i = 0; i < outputLayerNeurons.size(); ++i)
			outputValues.push_back(outputLayerNeurons[i]->GetValue());
		lastPredictionValues = outputValues;
		return (outputValues);
	}

	value_t					Network::GetCost(values_t expectedOutput)
	{
		if (costFunction != nullptr)
			return ((*costFunction)(GetOutputLayer()->GetNeurons(), expectedOutput));
		throw std::string("No cost function defined");
	}

	value_t					Network::GetDerivedCost(values_t expectedOutput, Neuron *outputNeuron)
	{
		if (costFunctionDerivative != nullptr)
			return ((*costFunctionDerivative)(GetOutputLayer()->GetNeurons(), expectedOutput, outputNeuron));
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

	void					Network::CalcNeuron(Neuron *neuron)
	{
		std::vector<Connection *>	connections;
		value_t 					currentDerivedValue;

		currentDerivedValue = 0;
		connections = neuron->GetOutputConnections();
		for (std::vector<Connection *>::size_type i = 0; i < connections.size(); ++i)
		{
			DEBUG_CHECK_VALUE(connections[i]->GetOutput()->GetDerivedError(), "Current derived value at output neuron");
			DEBUG_CHECK_VALUE(connections[i]->GetOutput()->GetDerivedValue(neuron), "Derived value w.r.t neuron (self)");
			DEBUG_CHECK_VALUE(currentDerivedValue, "Current derived value for self");
			currentDerivedValue += (connections[i]->GetOutput()->GetDerivedError() * connections[i]->GetOutput()->GetDerivedValue(neuron));
			DEBUG_CHECK_VALUE(currentDerivedValue, "New current derived value for self (" + std::to_string(connections[i]->GetOutput()->GetDerivedError()) + " * " + std::to_string(connections[i]->GetOutput()->GetDerivedValue(neuron)));
		}
		if (currentDerivedValue == INFINITY)
			currentDerivedValue = DBL_MAX;
		else if (currentDerivedValue == -INFINITY)
			currentDerivedValue = -DBL_MAX;
		neuron->SetDerivedError(currentDerivedValue);
		UpdateNeuronWeights(neuron);
	}

	void					Network::UpdateNeuronWeights(Neuron *neuron)
	{
		std::vector<Connection *>	connections;
		
		connections = neuron->GetInputConnections();
		for (std::vector<Connection *>::size_type i = 0; i < connections.size(); ++i)
		{
			weight_t	gradient;

			DEBUG_CHECK_VALUE(neuron->GetDerivedError(), "Current derived error");
			DEBUG_CHECK_VALUE(neuron->GetDerivedValue(connections[i]), "Derived value w.r.t connection");
			gradient = neuron->GetDerivedError() * neuron->GetDerivedValue(connections[i]);
			DEBUG_CHECK_VALUE(gradient, "gradient of weight");

			connections[i]->SetGradient(gradient);
			connections[i]->SetShouldUpdate(true);
		}
	}

	void					Network::Propagate(values_t goodValues, values_t derivedCost)
	{
		std::vector<Neuron *>		outputLayerNeurons;
		std::vector<Neuron *>		neurons;

		outputLayerNeurons = this->GetOutputLayer()->GetNeurons();
		for (std::vector<Neuron *>::size_type i = 0; i < outputLayerNeurons.size(); ++i)
		{
			DEBUG_CHECK_VALUE(derivedCost[i], "Derived cost");
			outputLayerNeurons[i]->SetDerivedError(derivedCost[i]);
			UpdateNeuronWeights(outputLayerNeurons[i]);
		}
		for (std::vector<Layer *>::size_type i = 0; i < layers.size() - 1; ++i)
		{
			neurons = layers[(layers.size() - 2) - i]->GetNeurons();
			for (std::vector<Neuron *>::size_type j = 0; j < neurons.size(); ++j)
			{
				if (!neurons[j]->IsBias())
					CalcNeuron(neurons[j]);
			}
		}
		this->UpdateWeights();
	}

	void					Network::Propagate(values_t goodValues)
	{
		std::vector<Neuron *>		outputLayerNeurons;
		values_t					derivedCost;

		outputLayerNeurons = this->GetOutputLayer()->GetNeurons();
		for (std::vector<Neuron *>::size_type i = 0; i < outputLayerNeurons.size(); ++i)
		{
			derivedCost.push_back(this->GetDerivedCost(goodValues, outputLayerNeurons[i]));
		}
		this->Propagate(goodValues, derivedCost);
	}

	void					Network::UpdateWeights()
	{
		std::vector<Connection *>		connections;
		value_t							maxGradient;
		bool							maxGradientFound = false;

		connections = Connection::GetConnections();
		if (normalizedGradient)
		{
			for (std::vector<Connection *>::size_type i = 0; i < connections.size(); ++i)
			{
				if (connections[i]->ShouldUpdate())
				{
					value_t gradient;

					gradient = connections[i]->GetGradient();
					if (maxGradientFound == false || ABS(gradient) > maxGradient)
					{
						maxGradient = ABS(gradient);
						maxGradientFound = true;
					}
				}
			}
			if (maxGradient < 1)
				maxGradient = 1;
		}
		for (std::vector<Connection *>::size_type i = 0; i < connections.size(); ++i)
		{
			if (connections[i]->ShouldUpdate())
			{
				value_t gradient;

				gradient = connections[i]->GetGradient();
				if (normalizedGradient)
				{
					gradient = gradient / maxGradient;
					//std::cout << maxGradient << std::endl;
					DEBUG_CHECK_VALUE(gradient, "gradient value after normalization");
				}
				gradient = learningRate * gradient;
				DEBUG_CHECK_VALUE(gradient, "gradient value after learning rate");

				if (weightPenality != 0)
				{
					if (connections[i]->GetWeight() < -weightPenality)
						gradient = gradient * (weightPenality / -connections[i]->GetWeight());
					else if (connections[i]->GetWeight() > weightPenality)
						gradient = gradient * (weightPenality / connections[i]->GetWeight());
				}
				
				if (gradientClipping != 0)
				{
					if (gradient > gradientClipping)
						gradient = gradientClipping;
					else if (gradient < -gradientClipping)
						gradient = -gradientClipping;
				}
				connections[i]->SetWeight(connections[i]->GetWeight() - gradient);
				connections[i]->SetShouldUpdate(false);
			}
		}
	}

	void					Network::SaveTo(std::string fileName)
	{
		std::ofstream file;
		file.open(fileName);
		file << VYN_NEURALNETWORK_STRING << std::endl;
		file << VYN_NEURALNETWORK_VERSION << std::endl;
		file << layers.size() << std::endl;
		for (std::vector<Layer *>::size_type i = 0; i < layers.size(); ++i)
		{
			std::vector<Neuron *> neurons;
			neurons = layers[i]->GetNeurons();
			for (std::vector<Neuron *>::size_type j = 0; j < neurons.size(); ++j)
			{
				file << neurons[j]->GetActivationFunctionId() << " ";
			}
			file << std::endl;
		}
		file << this->GetCostFunctionId() << std::endl;

		std::vector<Connection *>	connections;

		connections = Connection::GetConnections();
		for (std::vector<Connection *>::size_type i = 0; i < connections.size(); ++i)
		{
			file << connections[i]->GetWeight() << " ";
		}
		file.close();
	}
}