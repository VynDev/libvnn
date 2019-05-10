#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include <sstream>
#include <float.h>

#include "Types.h"
#include "Utils.h"
#include "Functions.h"

#define VYN_NEURALNETWORK_STRING "VYN_NEURALNETWORK"
#define VYN_NEURALNETWORK_VERSION "0.0.0"

//#define DEBUG // Comment this line to disable debug mode
#ifdef DEBUG
	#define DEBUG_CHECK_VALUE(x, msg) debug::CheckValue(x, std::string(__FUNCTION__) + std::string(": ") + msg)
#else
	#define DEBUG_CHECK_VALUE(x, msg)
#endif

namespace Vyn
{
	namespace NeuralNetwork
	{
		typedef struct				TrainingParameters_s
		{
			std::vector<Values>		trainingSetInputs;
			std::vector<Values>		trainingSetOutputs;
			std::vector<Values>		validationSetInputs;
			std::vector<Values>		validationSetOutputs;
			std::stringstream 		*trainingCsv = nullptr;
			std::stringstream 		*validationCsv = nullptr;

		}							TrainingParameters_t;

		class Network {

		/*
		**	Global neural network
		*/
		private:

			

			Layers		layers;
			Connections	connections;

			Value		(*costFunction)(const Neurons &, const Values &) = nullptr;
			Value		(*costFunctionDerivative)(const Neurons &, const Values &, Neuron *) = nullptr;
			int			costFunctionId = 0;
			Values		lastOutputValues;

		public:

			Network();


			Layer						*GetInputLayer() const;
			Layer						*GetOutputLayer() const;
			const Layers				&GetLayers() const {return (layers);};
			const Connections			&GetConnections() const {return (connections);};

			void						AddConnection(Connection *newConnection) {connections.push_back(newConnection);};
			void						RandomizeConnectionsWeight();

			void						AddLayer(Layer *layer);
			void						AddLayer(int nbNeuron, int neuronType = 0, int weightInitializationFunctionId = 0, int nbBias = 0);
			
			const Values						&Predict(Values const &inputs);

			void						SaveTo(std::string fileName);

		/*
		**	Back propagation
		*/

		private:

			Value						learningRate = 0.1;
			Value						gradientClipping = 0;
			bool						earlyStoppingEnabled = false;
			Value						errorPropagationLimit = 0;


			void						UpdateWeights();

		public:

			Value						GetLearningRate() const {return (learningRate);};
			void						SetLearningRate(Value newValue) {learningRate = newValue;};

			Value						GetCost(Values const &expectedOutput);
			Value						GetDerivedCost(Values const &expectedOutput, Neuron *outputNeuron);
			int							GetCostFunctionId() const {return (costFunctionId);};

			void						SetCostFunction(int functionId);
			void						SetCostFunction(Value (*f)(const Neurons &, const Values &)) {costFunctionId = 0; costFunction = f;};
			void						SetCostFunctionDerivative(Value (*f)(const Neurons &, const Values &, Neuron *)) {costFunctionId = 0; costFunctionDerivative = f;};

			void						SetGradientClipping(Value newValue) {gradientClipping = (newValue < 0 ? -newValue : newValue);};
			void						EnableEarlyStopping(bool newValue) {earlyStoppingEnabled = newValue;};
			void						SetErrorPropagationLimit(Value newValue) {errorPropagationLimit = newValue;};

			void						Fit(TrainingParameters_t parameters, int batchSize, int nbIteration);
			Value						TrainBatch(Network *network, std::vector<Values> &inputs, std::vector<Values> &expectedOutputs, int batchSize, int i);
			void						Propagate(Values const &goodValues);
			void						Propagate(Values const &goodValues, Values const &derivedCost);
			void						UpdateNeuronWeights(Neuron *neuron);
			void						CalcNeuron(Neuron *neuron);

		};
	}
}
#endif