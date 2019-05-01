/*
* @Author: Vyn
* @Date:   2019-03-16 18:09:48
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-01 19:21:23
*/

#include "Utils.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		namespace debug
		{
			void	CheckValue(value_t value, std::string msg)
			{
				if (std::isnan(value) || std::isinf(value))
				{
					std::cout << "[DEBUG] " << msg << ": " << value;
					throw msg + std::string(": ") + std::to_string(value);
				}
			};
		}
	}
}