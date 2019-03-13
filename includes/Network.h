#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <sstream>

#include "types.h"
#include "costfunctions.h"

namespace vyn::neuralnetwork {

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
		value_t						GetLearningRate() const {return (learningRate);};

		void						SetLearningRate(value_t newValue) {learningRate = newValue;};
		void						SetCostFunction(int functionId);
		void						SetCostFunction(value_t (*f)(std::vector<Neuron *>, values_t)) {costFunction = f;};
		void						SetCostFunctionDerivative(value_t (*f)(std::vector<Neuron *>, values_t, Neuron *)) {costFunctionDerivative = f;};

		void						AddLayer(Layer *layer);
		void						AddLayer(int nbNeuron, int neuronType = 0, int nbBias = 0);
		
		std::vector<value_t>		Predict(std::vector<value_t> inputs);
		value_t						GetCost(values_t expectedOutput);
		value_t						GetDerivedCost(values_t expectedOutput, Neuron *outputNeuron);
		void						Fit(std::vector<std::vector<value_t>> inputs, std::vector<std::vector<value_t>> outputs, int batchSize, int nbIteration, std::stringstream *csv);
		void						Propagate(values_t goodValues);
		void						Propagate(values_t goodValues, values_t derivedCost);


		void						UpdateNeuronWeights(Neuron *neuron);
		void						CalcNeuron(Neuron *neuron);

	};
}
#endif