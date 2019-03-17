/*
* @Author: Vyn
* @Date:   2019-02-01 12:36:17
* @Last Modified by:   Vyn
* @Last Modified time: 2019-03-15 14:02:18
*/

#include <iostream>
#include <cmath>

#include "Network.h"
#include "Neuron.h"
#include "Connection.h"

namespace vyn::neuralnetwork {

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
			value_t		inputNeuronValue;
			weight_t	weight;

			inputNeuronValue = inputs[i]->GetInput()->GetValue();
			weight = inputs[i]->GetWeight();
			DEBUG_CHECK_VALUE(inputNeuronValue, "ComputeValue: input neuron value");
			DEBUG_CHECK_VALUE(weight, "ComputeValue: weight value");
			newValue += inputNeuronValue * weight;
		}
		this->rawValue = newValue;
		this->value = newValue;
	}

	void	Neuron::ActivateFunction()
	{
		if (activationFuntion != nullptr)
		{
			this->value = (*activationFuntion)(this, rawValue);
		}
		else
		{
			throw std::string("No activation function for neuron");
		}
	}

	void		Neuron::SetActivationFunction(int id)
	{
		activationFunctionId = id;
		if (id == NEURON_FUNCTION_SIGMOID)
		{
			activationFuntion = &Sigmoid;
			activationFuntionDerivative = &SigmoidDerivative;
		}
		if (id == NEURON_FUNCTION_SOFTMAX)
		{
			activationFuntion = &Softmax;
			activationFuntionDerivative = &SoftmaxDerivative;
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
}