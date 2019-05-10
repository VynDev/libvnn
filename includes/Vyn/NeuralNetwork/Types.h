#ifndef TYPES_H
#define TYPES_H

#include <vector>

namespace Vyn
{
	namespace NeuralNetwork
	{
		typedef double Value;
		typedef double Weight;

		class Connection;
		class Neuron;
		class Layer;
		class Network;
		class Population;

		typedef std::vector<Value> Values;
		typedef std::vector<Connection *> Connections;
		typedef std::vector<Neuron *> Neurons;
		typedef std::vector<Layer *> Layers;
	}
}

#endif