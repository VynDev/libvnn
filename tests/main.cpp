/*
* @Author: Vyn
* @Date:   2019-04-18 15:01:42
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-13 16:03:36
*/

#include <iostream>

#include "vtest/vtest.hpp"
#include "Vyn/NeuralNetwork.h"

TEST_REGISTER(NEURON);
TEST_REGISTER(CONNECTION);
TEST_REGISTER(LAYER);
TEST_REGISTER(NETWORK);

int		main(void)
{
	try
	{
		srand(time(NULL));
		TEST_EXECUTE(CONNECTION);
		TEST_EXECUTE(NEURON);
		TEST_EXECUTE(LAYER);
		TEST_EXECUTE(NETWORK);
	}
	catch (std::string e)
	{
		std::cout << e << std::endl;
	}
	return (0);
}