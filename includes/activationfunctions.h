#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include "types.h"

class Neuron;

value_t		Sigmoid(Neuron *, value_t x);
value_t		SigmoidDerivative(Neuron *, value_t x);

#endif