#ifndef TYPES_H
#define TYPES_H

#include <vector>

namespace vyn::neuralnetwork {
	class Neuron;
	class Connection;
	class Layer;
	class Network;

	typedef double value_t;
	typedef double weight_t;
	typedef std::vector<value_t> values_t;
}

#endif