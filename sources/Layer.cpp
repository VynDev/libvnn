/*
* @Author: Vyn
* @Date:   2019-02-02 11:29:33
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-04 11:35:13
*/

#include <iostream>

#include "Layer.h"
#include "Neuron.h"

Layer::Layer(int nbNeuron, int functionId, int nbBias)
{
	++nbLayer;
	this->nbBias += nbBias;
	(void)functionId;
	for (int i = 0; i < nbNeuron; ++i)
	{
		AddNeuron(NEURON_FUNCTION_SIGMOID);
	}
	for (int i = 0; i < nbBias; ++i)
	{
		AddBias();
	}
}

void					Layer::AddNeuron(int functionId)
{
	Neuron	*neuron;

	neuron = new Neuron(functionId);
	neuron->SetParentLayer(this);
	neurons.push_back(neuron);
}

void					Layer::AddBias()
{
	Neuron	*neuron;

	neuron = new Neuron(NEURON_FUNCTION_BIAS);
	neuron->SetParentLayer(this);
	neurons.push_back(neuron);
}

void					Layer::ConnectTo(Layer *layer)
{
	std::vector<Neuron *> toNeurons;
	toNeurons = layer->GetNeurons();
	for (std::vector<Neuron *>::size_type i = 0; i < neurons.size(); ++i)
	{
		for (std::vector<Neuron *>::size_type j = 0; j < toNeurons.size(); ++j)
		{
			if (toNeurons[j]->IsBias() == false)
				neurons[i]->ConnectTo(toNeurons[j]);
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
	for (std::vector<Neuron *>::size_type i = 0; i != neurons.size(); ++i)
	{
		if (!neurons[i]->IsBias())
			neurons[i]->ComputeValue();
	}
}

std::vector<value_t>	Layer::GetValues() const
{
	std::vector<value_t> values;
	for (std::vector<Neuron *>::size_type i = 0; i != neurons.size(); ++i)
	{
		values.push_back(neurons[i]->GetValue());
	}
	return values;
}

void					Layer::Describe(bool showNeuronsValue)
{
	std::cout << "--- Layer ID: " << this->id << std::endl;
	std::cout << "Type: : " << this->type << std::endl;
	std::cout << "Number of neurons: " << this->neurons.size() << std::endl;
	if (showNeuronsValue)
	{
		std::vector<value_t> values;

		values = this->GetValues();
		for (std::vector<value_t>::size_type i = 0; i != values.size(); ++i)
		{
			std::cout << values[i] << std::endl;
		}
	}
	std::cout << "---" << std::endl;
}

int	Layer::nbLayer = 0;