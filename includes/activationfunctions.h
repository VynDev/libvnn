#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#define NEURON_FUNCTION_BIAS 0
#define NEURON_FUNCTION_SIGMOID 1
#define NEURON_FUNCTION_SOFTMAX 2

#include "types.h"

class Neuron;

value_t		Sigmoid(Neuron *, value_t x);
value_t		SigmoidDerivative(Neuron *, value_t x);

value_t		Softmax(Neuron *, value_t x);
value_t		SoftmaxDerivative(Neuron *, value_t x);

#endif