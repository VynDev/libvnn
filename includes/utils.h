#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#include "Network.h"
#include "Layer.h"
#include "Connection.h"
#include "types.h"

namespace vyn::neuralnetwork
{

	namespace scale
	{
		void	MinMax(std::vector<std::vector<value_t>> &inputs);
	}

	Network		Load(std::string fileName);

	// DEBUG

	namespace debug
	{
		void	CheckValue(value_t value, std::string msg);
	}
}

#endif