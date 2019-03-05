/*
* @Author: Vyn
* @Date:   2019-02-01 12:36:17
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-04 17:13:28
*/

#include <iostream>
#include <cmath>

#include "Neuron.h"

#include "Connection.h"

Neuron::Neuron()
{
	++nbNeuron;
}

Neuron::Neuron(int type)
{
	Neuron();
	SetActivationFunction(type);
	if (type == NEURON_FUNCTION_BIAS)
		SetBias(true);
}

void	Neuron::ConnectTo(Neuron *neuron)
{
	Connection	*newConnection;
	Neuron		*fromNeuron;
	Neuron		*toNeuron;

	newConnection = new Connection;
	fromNeuron = this;
	toNeuron = neuron;
	
	newConnection->SetInput(fromNeuron);
	newConnection->SetOutput(toNeuron);

	outputs.push_back(newConnection);
	toNeuron->AddInputConnection(newConnection);
}

void	Neuron::AddInputConnection(Connection *newConnection)
{
	inputs.push_back(newConnection);
}

void	Neuron::ComputeValue()
{
	value_t	newValue;

	newValue = 0;
	for (std::vector<Connection *>::size_type i = 0; i != inputs.size(); ++i)
	{
		Neuron		*inputNeuron;
		weight_t	weight;

		inputNeuron = inputs[i]->GetInput();
		weight = inputs[i]->GetWeight();
		newValue += inputNeuron->GetValue() * weight;
	}
	if (activationFuntion != nullptr)
	{
		this->rawValue = newValue;
		this->value = (*activationFuntion)(this, newValue);
	}
	else
	{
		throw std::string("No activation function for neuron");
	}

}

void		Neuron::SetActivationFunction(int id)
{
	if (id == NEURON_FUNCTION_SIGMOID)
	{
		this->SetActivationFunction(&Sigmoid);
		this->SetActivationFunctionDerivative(&SigmoidDerivative);
	}
}

value_t		Neuron::GetDerivedValue(Connection *connection)
{
	return ((*activationFuntionDerivative)(this, GetRawValue()) * connection->GetInput()->GetValue());
}

value_t		Neuron::GetDerivedValue(Neuron *neuron)
{
	if (neuron->IsBias())
		throw std::string("Trying to derive with respect to bias neuron");
	for (std::vector<Connection *>::size_type i = 0; i < inputs.size(); ++i)
	{
		if (inputs[i]->GetInput() == neuron)
			return ((*activationFuntionDerivative)(this, GetRawValue()) * inputs[i]->GetWeight());
	}
	throw std::string("Can't derive neuron with respect to neuron");
}

int	Neuron::nbNeuron = 0;