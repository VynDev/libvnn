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

		}							TrainingParameters;

		class Network {

		/*
		**	Global neural network
		*/
		private:

			Layers		layers;
			Connections	connections;

			Value		(*costFunction)(const Values &, const Values &) = nullptr;
			Value		(*costFunctionDerivative)(const Values &, const Values &, int) = nullptr;
			int			costFunctionId = 0;
			Values		lastOutputValues;
			Values		tmpDerivedCost;

		public:

			Network();


			Layer						*GetInputLayer() const;
			Layer						*GetOutputLayer() const;
			const Layers				&GetLayers() const {return (layers);};
			const Connections			&GetConnections() const {return (connections);};

			void						AddConnection(Connection *newConnection) {connections.push_back(newConnection);};
			void						RandomizeConnectionsWeight();

			void						AddLayer(Layer *layer);
			void						AddLayer(int nbNeuron, int neuronType = 0, int weightInitializationFunctionId = 1);
			
			const Values				&Predict(Values const &inputs);

			void						SaveTo(std::string fileName);

		/*
		**	Back propagation
		*/

		private:

			Value						learningRate = 0.1;
			Value						gradientClipping = 0;
			bool						earlyStoppingEnabled = false;
			bool						shuffleEnabled = false;
			Value						errorPropagationLimit = 0;

			std::stringstream 			trainingCsv;
			std::stringstream 			validationCsv;


			void						UpdateWeights();

		public:

			Value						GetLearningRate() const {return (learningRate);};
			void						SetLearningRate(Value newValue) {learningRate = newValue;};

			Value						GetCost(Values const &expectedOutput);
			Value						GetDerivedCost(Values const &expectedOutput, int neuronIndex);
			int							GetCostFunctionId() const {return (costFunctionId);};

			void						SetCostFunction(int functionId);
			void						SetCostFunction(Value (*f)(const Values &, const Values &)) {costFunctionId = 0; costFunction = f;};
			void						SetCostFunctionDerivative(Value (*f)(const Values &, const Values &, int)) {costFunctionId = 0; costFunctionDerivative = f;};

			void						SetGradientClipping(Value newValue) {gradientClipping = (newValue < 0 ? -newValue : newValue);};
			void						EnableEarlyStopping(bool newValue) {earlyStoppingEnabled = newValue;};
			void						EnableShuffle(bool newValue) {shuffleEnabled = newValue;};
			void						SetErrorPropagationLimit(Value newValue) {errorPropagationLimit = newValue;};

			void						Fit(TrainingParameters parameters, int batchSize, int nbIteration);
			Value						TrainBatch(Network *network, std::vector<Values> &inputs, std::vector<Values> &expectedOutputs, int batchSize, int i, std::vector<int> &indexes);
			void						Propagate(Values const &goodValues);
			void						Propagate(Values const &goodValues, Values const &derivedCost);
			void						UpdateNeuronWeights(Neuron *neuron);
			void						CalcNeuron(Neuron *neuron);

			void						SaveTrainingCsv(std::string path);
			void						SaveValidationCsv(std::string path);

		};
	}
}
#endif