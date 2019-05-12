/*
* @Author: Vyn
* @Date:   2019-02-02 11:29:33
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-12 15:48:10
*/

#include <iostream>

#include "Layer.h"
#include "Neuron.h"

#define ABS(x) (x < 0 ? -x : x)

namespace Vyn
{
	namespace NeuralNetwork
	{
		Layer::Layer() {}
		
		Layer::Layer(int nbNeuron, int functionId, int weightInitializationFunctionId)
		{
			SetWeightInitialization(weightInitializationFunctionId);
			++nbLayer;
			for (int i = 0; i < nbNeuron; ++i)
			{
				AddNeuron(functionId);
			}
		}

		void					Layer::AddNeuron(int functionId)
		{
			Neuron	*neuron;

			if (functionId == NEURON_FUNCTION_BIAS)
				this->nbBias += 1;
			neuron = new Neuron(functionId);
			neuron->SetParentLayer(this);
			neurons.push_back(neuron);
		}

		void					Layer::AddBias()
		{
			Neuron	*neuron;

			this->nbBias += 1;
			neuron = new Neuron(NEURON_FUNCTION_BIAS);
			neuron->SetParentLayer(this);
			neurons.push_back(neuron);
		}

		void					Layer::ConnectTo(Layer *layer)
		{
			Neurons toNeurons;
			toNeurons = layer->GetNeurons();
			for (Neurons::size_type i = 0; i < neurons.size(); ++i)
			{
				for (Neurons::size_type j = 0; j < toNeurons.size(); ++j)
				{
					if (toNeurons[j]->IsBias() == false)
					{
						neurons[i]->ConnectTo(toNeurons[j]);
					}
				}
			}
			layer->AddInput(this);
			output = layer;
		}

		void					Layer::AddInput(Layer *layer)
		{
			input = layer;
		}

		void					Layer::ComputeValues()
		{
			if (twoStepActivationEnabled)
			{
				for (Neurons::size_type i = 0; i != neurons.size(); ++i)
				{
					if (!neurons[i]->IsBias())
						neurons[i]->ComputeValue();
				}
				for (Neurons::size_type i = 0; i != neurons.size(); ++i)
				{
					if (!neurons[i]->IsBias())
						neurons[i]->ActivateFunction();
				}
			}
			else
			{
				for (Neurons::size_type i = 0; i != neurons.size(); ++i)
				{
					if (!neurons[i]->IsBias())
					{
						neurons[i]->ComputeValue();
						neurons[i]->ActivateFunction();
					}
				}
			}
		}

		Values	Layer::GetValues() const
		{
			Values values;

			values.reserve(neurons.size());
			for (Neurons::size_type i = 0; i != neurons.size(); ++i)
			{
				values.push_back(neurons[i]->GetValue());
			}
			return (values);
		}

		void					Layer::Describe(bool showNeuronsValue)
		{
			std::cout << "--- Layer ID: " << this->id << std::endl;
			std::cout << "Type: : " << this->type << std::endl;
			std::cout << "Number of neurons: " << this->neurons.size() << std::endl;
			if (showNeuronsValue)
			{
				Values values;

				values = this->GetValues();
				for (Values::size_type i = 0; i != values.size(); ++i)
				{
					std::cout << values[i] << std::endl;
				}
			}
			std::cout << "---" << std::endl;
		}

		void					Layer::SetWeightInitialization(int initializationId)
		{
			if (initializationId == WEIGHT_INIT_0)
				weightInitializationFunction = &weightInitialization0;
			if (initializationId == WEIGHT_INIT_1)
				weightInitializationFunction = &weightInitialization1;
		}
		int	Layer::nbLayer = 0;
	}
}