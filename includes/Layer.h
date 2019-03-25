#ifndef LAYER_H
#define LAYER_H

#define LAYER_UNCONNECTED 0
#define LAYER_INPUT 1
#define LAYER_HIDDEN 2
#define LAYER_OUTPUT 3

#include <vector>
#include "types.h"

namespace vyn::neuralnetwork {

	class Layer {

	private:

		static int	nbLayer;

		const int										id = nbLayer;

		std::vector<vyn::neuralnetwork::Neuron *>		neurons;
		Layer											*input = nullptr;
		Layer											*output = nullptr;
		int												type = LAYER_UNCONNECTED;
		int												nbBias = 0;
		bool											twoStepActivationEnabled = true;

	public:

		Layer(int nbNeuron, int functionId, int nbBias = 0);

		void											ConnectTo(Layer *layer);
		void											AddInput(Layer *layer);
		void											AddNeuron(int functionId);
		void											AddBias();
		void											ComputeValues();
		void											Describe(bool showNeuronsValue = false);
		void											EnableTwoStepActivation(bool newValue) {twoStepActivationEnabled = newValue;};

		std::vector<vyn::neuralnetwork::Neuron *>		GetNeurons() const {return (neurons);};
		std::vector<value_t>							GetValues() const;
		int												GetBiasCount() const {return (nbBias);};
	};
}
#endif