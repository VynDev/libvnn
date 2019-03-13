#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "types.h"

namespace vyn::neuralnetwork {
	namespace scale {
		void	MinMax(std::vector<std::vector<value_t>> &inputs);
	}
}

#endif