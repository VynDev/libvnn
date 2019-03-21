#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <sstream>
#include <float.h>

#include "types.h"
#include "utils.h"
#include "costfunctions.h"

#define VYN_NEURALNETWORK_STRING "VYN_NEURALNETWORK"
#define VYN_NEURALNETWORK_VERSION "0.0.0"

//#define DEBUG // Comment this line to disable debug mode
#ifdef DEBUG
	#define DEBUG_CHECK_VALUE(x, msg) debug::CheckValue(x, std::string(__FUNCTION__) + std::string(": ") + msg)
#else
	#define DEBUG_CHECK_VALUE(x, msg)
#endif

namespace vyn::neuralnetwork {

	class Network {

	private:

		std::vector<Layer *>		layers;

		value_t						learningRate = 0.1;
		value_t						gradientClipping = 0;
		value_t						weightPenality = 0;
		bool						normalizedGradient = false;

		value_t						(*costFunction)(std::vector<Neuron *>, values_t) = nullptr;
		value_t						(*costFunctionDerivative)(std::vector<Neuron *>, values_t, Neuron *) = nullptr;
		int							costFunctionId = 0;
		std::vector<value_t>		lastPredictionValues;

		void						UpdateWeights();

	public:

		Network();

		/*
		**	Global neural network functions
		*/

		Layer						*GetInputLayer() const;
		Layer						*GetOutputLayer() const;
		std::vector<Layer *>		GetLayers() const {return (layers);};

		void						AddLayer(Layer *layer);
		void						AddLayer(int nbNeuron, int neuronType = 0, int nbBias = 0);
		
		std::vector<value_t>		Predict(std::vector<value_t> inputs);

		void						SaveTo(std::string fileName);

		/*
		**	Supervised learning functions
		*/

		value_t						GetLearningRate() const {return (learningRate);};
		int							GetCostFunctionId() const {return (costFunctionId);};
		value_t						GetCost(values_t expectedOutput);
		value_t						GetDerivedCost(values_t expectedOutput, Neuron *outputNeuron);

		void						SetLearningRate(value_t newValue) {learningRate = newValue;};
		void						SetGradientClipping(value_t newValue) {gradientClipping = (newValue < 0 ? -newValue : newValue);};
		void						SetWeightPenality(value_t newValue) {weightPenality = (newValue < 0 ? -newValue : newValue);};
		void						SetNormalizedGradient(bool newValue) {normalizedGradient = newValue;};

		void						SetCostFunction(int functionId);
		void						SetCostFunction(value_t (*f)(std::vector<Neuron *>, values_t)) {costFunctionId = 0; costFunction = f;};
		void						SetCostFunctionDerivative(value_t (*f)(std::vector<Neuron *>, values_t, Neuron *)) {costFunctionId = 0; costFunctionDerivative = f;};

		void						Fit(std::vector<std::vector<value_t>> inputs, std::vector<std::vector<value_t>> outputs, int batchSize, int nbIteration, std::stringstream *csv);
		void						Propagate(values_t goodValues);
		void						Propagate(values_t goodValues, values_t derivedCost);
		void						UpdateNeuronWeights(Neuron *neuron);
		void						CalcNeuron(Neuron *neuron);

	};
}
#endif