#ifndef CONNECTION_H
#define CONNECTION_H

#include <vector>
#include <cstdlib>

#include "Types.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		class Connection {

		/*
		**	Global neural network
		*/
		private:


			Neuron								*input = nullptr;
			Neuron								*output = nullptr;
			Weight							weight = 0;

		public:

			Connection();

			void								SetInput(Neuron *neuron) {input = neuron;};
			void								SetOutput(Neuron *neuron) {output = neuron;};
			void								SetWeight(Weight newWeight) {weight = newWeight;};


			Neuron								*GetInput() const {return (input);};
			Neuron								*GetOutput() const {return (output);};
			Weight								GetWeight() const {return (weight);};
		/*
		**	Back propagation
		*/
		private:

			Value								gradient = 0;
			bool								shouldUpdate = false;

		public:

			void								SetGradient(Value newGradient) {gradient = newGradient;};
			void								SetShouldUpdate(bool a) {shouldUpdate = a;};

			Weight								GetGradient() const {return (gradient);};
			bool								ShouldUpdate() const {return (shouldUpdate);};
		};
	}
}
#endif