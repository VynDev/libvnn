/*
* @Author: Vyn
* @Date:   2019-02-01 12:36:17
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 13:36:44
*/

#include <iostream>
#include <cmath>

#include "Network.h"
#include "Neuron.h"
#include "Connection.h"

namespace Vyn
{

	namespace NeuralNetwork
	{

		/** Base constructor, shouldn't be used unless you know what you are doing */
		Neuron::Neuron()
		{
			++nbNeuron;
		}

		/**
		*	@param type		The type of the neuron
		*/
		Neuron::Neuron(int type)
		{
			Neuron();
			SetActivationFunction(type);
			if (type == NEURON_FUNCTION_BIAS)
				SetBias(true);
		}

		/**
		*	Connect the neuron to the specified neuron (first parameter)
		*
		*	@param neuron	The neuron to connect
		*/
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
			newConnection->SetWeight(0);
			if (neuron->GetParentLayer())
				newConnection->SetWeight(neuron->GetParentLayer()->NewWeightValue());
			if (neuron->GetParentLayer() && neuron->GetParentLayer()->GetParentNetwork())
				neuron->GetParentLayer()->GetParentNetwork()->AddConnection(newConnection);

			outputs.push_back(newConnection);
			toNeuron->AddInputConnection(newConnection);
		}

		/**
		*	Add an input connection to the neuron
		*
		*	@param newConnection	The connection to add as input
		*/
		void	Neuron::AddInputConnection(Connection *newConnection)
		{
			inputs.push_back(newConnection);
		}

		/**
		*	Compute the **raw** value of the neuron: 
		*	The sum of all the incoming connections (product of previous neurons and weights)  
		*	This step happens just **before** the activation function (ActivateFunction)
		*/
		void	Neuron::ComputeValue()
		{

			this->rawValue = 0;
			for (Connections::size_type i = 0; i != inputs.size(); ++i)
			{
				this->rawValue += inputs[i]->GetInput()->GetValue() * inputs[i]->GetWeight();
			}
			this->derivedRawValue = 0;
		}

		/**
		*	Compute the **real** value of the neuron: 
		*	It takes the **raw** value from the previous step (ComputeValue) and pass it to the activation function
		*/
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

		/**
		*	Set the neuron activation function by ID:  
		*	The lib provides some activation functions (Sigmoid, Softmax)
		*	
		*	@param id	The activation function ID:  
		*				NEURON_FUNCTION_SIGMOID  
		*				NEURON_FUNCTION_SOFTMAX
		*/
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

		/**
		*	Get the derived value of this neuron with respect to *connection* (first parameter)
		*
		*	@param connection	The connection to derive w.r.t
		*/
		Value		Neuron::GetDerivedValue(Connection *connection)
		{
			if (derivedRawValue == 0)
				derivedRawValue = (*activationFuntionDerivative)(this, GetRawValue());
			return (derivedRawValue * connection->GetInput()->GetValue());
		}

		/**
		*	Get the derived value of this neuron with respect to *neuron* (first parameter)
		*
		*	@param neuron	The neuron to derive w.r.t
		*/
		Value		Neuron::GetDerivedValue(Neuron *neuron)
		{
			if (neuron->IsBias())
				throw std::string("Trying to derive with respect to bias neuron");
			for (Connections::size_type i = 0; i < inputs.size(); ++i)
			{
				if (inputs[i]->GetInput() == neuron)
				{
					if (derivedRawValue == 0)
						derivedRawValue = (*activationFuntionDerivative)(this, GetRawValue());
					return (derivedRawValue * inputs[i]->GetWeight());
				}
			}
			throw std::string("Can't derive neuron with respect to neuron");
		}
		int	Neuron::nbNeuron = 0;
	}
}