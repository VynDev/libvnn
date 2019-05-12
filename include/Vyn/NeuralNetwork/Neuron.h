#ifndef NEURON_H
#define NEURON_H

#include <vector>

#include "Layer.h"
#include "Types.h"
#include "Functions.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		/** 
		*	Neuron component of the neural network
		*/

		class Neuron {

		/*
		**	Global neural network
		*/
		private:

			static int					nbNeuron;

			const int					id = nbNeuron;
			Connections	inputs;
			Connections	outputs;
			Value						rawValue;
			Value						value = 1;
			Value						(*activationFuntion)(Neuron *, Value) = nullptr;
			Value						(*activationFuntionDerivative)(Neuron *, Value) = nullptr;
			int							activationFunctionId = 0;
			bool						isBias = false;

			Layer						*parentLayer = nullptr;
			friend void					Layer::AddNeuron(int functionId);
			friend void					Layer::AddBias();
			void						SetParentLayer(Layer *layer) {parentLayer = layer;};

			Value						derivedRawValue = 0;

		public:

			Neuron();
			Neuron(int type);
			
			void						ConnectTo(Neuron *neuron);
			void						AddInputConnection(Connection *newConnection);
			void						ComputeValue();
			void						ActivateFunction();
			void						SetValue(Value newValue) {value = newValue;};
			void						SetBias(bool newValue) {isBias = newValue;};

			void						SetActivationFunction(int id);
			void						SetActivationFunction(Value (*f)(Neuron *, Value)) {activationFunctionId = 0; activationFuntion = f;};

			bool						IsBias() const {return (isBias);};

			int							GetActivationFunctionId() const {return (activationFunctionId);};
			const Connections 			&GetInputConnections() const {return (inputs);};
			const Connections 			&GetOutputConnections() const {return (outputs);};
			int							GetId() const {return (id);};
			Layer						*GetParentLayer() const {return (parentLayer);};

			Value						GetValue() const {return (value);};
			Value						GetRawValue() const {return (rawValue);};

			Value						(*GetActivationFunction(void))(Neuron *, Value) {return (activationFuntion);};
		/*
		**	Back propagation
		*/
		private:

			Value						derivedError = 0;

		public:

			void						SetDerivedError(Value newValue) {derivedError = newValue;};

			Value						GetDerivedValue(Connection *connection); // derived with respect to weight
			Value						GetDerivedValue(Neuron *neuron); // derived with respect to neuron
			Value						GetDerivedError() const {return (derivedError);};

			void						SetActivationFunctionDerivative(Value (*f)(Neuron *, Value)) {activationFunctionId = 0; activationFuntionDerivative = f;};

		};
	}
}

#endif