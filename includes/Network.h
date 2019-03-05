#ifndef NETWORK_H
#define NETWORK_H

#define COST_FUNCTION_MSE 0

#include <vector>

#include "types.h"
#include "costfunctions.h"

class Layer;
class Neuron;

class Network {

private:

	std::vector<Layer *>		layers;

	value_t						learningRate = 0.1;
	value_t						(*costFunction)(std::vector<Neuron *>, values_t) = nullptr;
	value_t						(*costFunctionDerivative)(std::vector<Neuron *>, values_t, Neuron *) = nullptr;
	std::vector<value_t>		lastPredictionValues;

	void						UpdateWeights();

public:

	Network();

	Layer						*GetInputLayer() const;
	Layer						*GetOutputLayer() const;
	std::vector<Layer *>		GetLayers() const {return (layers);};

	void						SetCostFunction(int functionId);
	void						SetCostFunction(value_t (*f)(std::vector<Neuron *>, values_t)) {costFunction = f;};
	void						SetCostFunctionDerivative(value_t (*f)(std::vector<Neuron *>, values_t, Neuron *)) {costFunctionDerivative = f;};

	void						AddLayer(Layer *layer);
	void						AddLayer(int nbNeuron, int neuronType = 0, int nbBias = 0);
	
	std::vector<value_t>		Predict(std::vector<value_t> inputs);
	value_t						GetCost(values_t expectedOutput);
	value_t						GetDerivedCost(values_t expectedOutput, Neuron *outputNeuron);
	void						Propagate(value_t goodValue);

};

#endif