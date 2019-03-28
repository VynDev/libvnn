#ifndef CONNECTION_H
#define CONNECTION_H

#include <vector>
#include <cstdlib>

#include "types.h"

namespace vyn::neuralnetwork {

	class Connection {

	private:

		static int							nbConnection;
		static std::vector<Connection *>	connections;

		const int							id = nbConnection;
		Neuron								*input = nullptr;
		Neuron								*output = nullptr;
		weight_t							weight = ((value_t)rand() / (value_t)RAND_MAX < 0.5 ? (value_t)rand() / (value_t)RAND_MAX * -0.6 - 0.2 : (value_t)rand() / (value_t)RAND_MAX * 0.6 + 0.2); //0.3 + (((value_t)rand() / (value_t)RAND_MAX) * 0.7);
		value_t								gradient = 0;

		bool								shouldUpdate = false;

	public:

		Connection();

		void								SetInput(Neuron *neuron) {input = neuron;};
		void								SetOutput(Neuron *neuron) {output = neuron;};
		void								SetWeight(weight_t newWeight) {weight = newWeight;};
		void								SetGradient(value_t newGradient) {gradient = newGradient;};
		void								SetShouldUpdate(bool a) {shouldUpdate = a;};

		bool								ShouldUpdate() const {return (shouldUpdate);};
		int									GetId() const {return (id);};
		Neuron								*GetInput() const {return (input);};
		Neuron								*GetOutput() const {return (output);};
		weight_t							GetWeight() const {return (weight);};
		weight_t							GetGradient() const {return (gradient);};

		static std::vector<Connection *>	GetConnections() {return (connections);};

	};
}
#endif