#ifndef CONNECTION_H
#define CONNECTION_H

#include <vector>
#include <cstdlib>

#include "types.h"

class Neuron;

class Connection {

private:

	static int							nbConnection;
	static std::vector<Connection *>	connections;

	const int							id = nbConnection;
	Neuron								*input = nullptr;
	Neuron								*output = nullptr;
	weight_t							weight = (value_t)rand() / (value_t)RAND_MAX;
	weight_t							nextWeight = -100;

	bool								shouldUpdate = false;

public:

	Connection();

	void								SetInput(Neuron *neuron) {input = neuron;};
	void								SetOutput(Neuron *neuron) {output = neuron;};
	void								SetWeight(weight_t newWeight) {weight = newWeight;};
	void								SetNextWeight(weight_t newNextWeight) {nextWeight = newNextWeight;};
	void								SetShouldUpdate(bool a) {shouldUpdate = a;};

	bool								ShouldUpdate() const {return (shouldUpdate);};
	int									GetId() const {return (id);};
	Neuron								*GetInput() const {return (input);};
	Neuron								*GetOutput() const {return (output);};
	weight_t							GetWeight() const {return (weight);};
	weight_t							GetNextWeight() const {return (nextWeight);};

	static std::vector<Connection *>	GetConnections() {return (connections);};

};

#endif