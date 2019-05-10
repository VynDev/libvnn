#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#include "Network.h"
#include "Layer.h"
#include "Connection.h"
#include "Types.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		Network		*Load(std::string fileName);

		typedef struct				ScaleData_s
		{
			Value		min;
			Value		max;
			Value		mean;
		} 							ScaleData;
		std::vector<ScaleData>	MinMax(std::vector<Values> &inputs);
		std::vector<ScaleData>	MinMax(std::vector<Values> &inputs, std::vector<ScaleData> &scaleDatas);
		std::vector<ScaleData>	MeanNormalisation(std::vector<Values> &inputs);
		std::vector<ScaleData>	MeanNormalisation(std::vector<Values> &inputs, std::vector<ScaleData> &scaleDatas);

		void SaveScaleDatas(std::vector<ScaleData> scaleDatas, std::string fileName);
		std::vector<ScaleData>	LoadScaleDatas(std::string fileName);

		namespace debug
		{
			void	CheckValue(Value value, std::string msg);
		}
	}
}

#endif