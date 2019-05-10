/*
* @Author: Vyn
* @Date:   2019-03-13 12:53:26
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-10 15:23:45
*/

#include <iostream>
#include <sstream>
#include <fstream>
#include "Utils.h"

namespace Vyn
{
	namespace NeuralNetwork
	{
		Network		*Load(std::string fileName)
		{
			Network							*network;
			Layer							*layer;
			Connections		connections;

			std::ifstream 					file;
			std::string						line;
			std::stringstream				lineStream;
			std::string						elem;
			int								nbLayers;

			network = new Network;
			file.open(fileName);
			getline(file, line);
			std::cout << "Checking string: " << line << std::endl;
			if (line != VYN_NEURALNETWORK_STRING)
				throw std::string("String does not match");
			getline(file, line);
			std::cout << "Checking version: " << line << std::endl;
			if (line != VYN_NEURALNETWORK_VERSION)
				throw std::string("Version does not match");
			getline(file, line);
			std::cout << "Getting number of layers: " << line << std::endl;
			nbLayers = std::stoi(line);
			for (int i = 0; i < nbLayers; ++i)
			{
				getline(file, line);
				lineStream = std::stringstream(line);
				std::cout << "Creating new layer" << std::endl;
				layer = new Layer(0, 0, 0);
				while (getline(lineStream, elem, ' '))
				{
					std::cout << "Adding neuron with function ID: " << elem << std::endl;
					layer->AddNeuron(std::stoi(elem));
				}
				network->AddLayer(layer);
			}
			std::cout << "Loading cost function" << std::endl;
			getline(file, line);
			network->SetCostFunction(std::stoi(line));
			getline(file, line);
			lineStream = std::stringstream(line);
			connections = network->GetConnections();
			int	i = 0;
			std::cout << "Loading weights" << std::endl;
			while (getline(lineStream, elem, ' '))
			{
				connections[i]->SetWeight(std::stod(elem));
				++i;
			}
			if (i != connections.size())
				throw std::string("The model is corrupted ?");
			std::cout << "Network fully loaded" << std::endl;
			return (network);
		}

		Value GetMean(std::vector<Values> &inputs, int i)
		{
			Value	total;

			total = 0;
			for (int j = 0; j < inputs.size(); ++j)
			{
				total += inputs[j][i];
			}
			return (total / inputs.size());
		}

		Value GetMin(std::vector<Values> &inputs, int i)
		{
			Value	min;

			min = inputs[0][i];
			for (int j = 0; j < inputs.size(); ++j)
			{
				if (inputs[j][i] < min)
					min = inputs[j][i];
			}
			return (min);
		}

		Value	GetMax(std::vector<Values> &inputs, int i)
		{
			Value	max;

			max = inputs[0][i];
			for (int j = 0; j < inputs.size(); ++j)
			{
				if (inputs[j][i] > max)
					max = inputs[j][i];
			}
			return (max);
		}

		std::vector<ScaleData> MinMax(std::vector<Values> &inputs)
		{
			std::vector<ScaleData>	scaleDatas;
			Values		firstLine;

			firstLine = inputs[0];
			for (int i = 0; i < firstLine.size(); ++i)
			{
				ScaleData		scaleData;
				scaleData.min = GetMin(inputs, i);
				scaleData.max = GetMax(inputs, i);
				scaleData.mean = GetMean(inputs, i);
				for (int j = 0; j < inputs.size(); ++j)
				{
					inputs[j][i] = (inputs[j][i] - scaleData.min) / (scaleData.max - scaleData.min);
				}
				scaleDatas.push_back(scaleData);
			}
			return (scaleDatas);
		}

		std::vector<ScaleData> MinMax(std::vector<Values> &inputs, std::vector<ScaleData> &scaleDatas)
		{
			Values		firstLine;

			firstLine = inputs[0];
			for (int i = 0; i < firstLine.size(); ++i)
			{
				for (int j = 0; j < inputs.size(); ++j)
				{
					inputs[j][i] = (inputs[j][i] - scaleDatas[i].min) / (scaleDatas[i].max - scaleDatas[i].min);
				}
			}
			return (scaleDatas);
		}

		std::vector<ScaleData> MeanNormalisation(std::vector<Values> &inputs)
		{
			std::vector<ScaleData>	scaleDatas;
			Values		firstLine;

			firstLine = inputs[0];
			for (int i = 0; i < firstLine.size(); ++i)
			{
				ScaleData		scaleData;
				scaleData.mean = GetMean(inputs, i);
				scaleData.min = GetMin(inputs, i);
				scaleData.max = GetMax(inputs, i);
				for (int j = 0; j < inputs.size(); ++j)
				{
					inputs[j][i] = (inputs[j][i] - scaleData.mean) / (scaleData.max - scaleData.min);
				}
				scaleDatas.push_back(scaleData);
			}
			return (scaleDatas);
		}

		std::vector<ScaleData> MeanNormalisation(std::vector<Values> &inputs, std::vector<ScaleData> &scaleDatas)
		{
			Values		firstLine;

			firstLine = inputs[0];
			for (int i = 0; i < firstLine.size(); ++i)
			{
				for (int j = 0; j < inputs.size(); ++j)
				{
					inputs[j][i] = (inputs[j][i] - scaleDatas[i].mean) / (scaleDatas[i].max - scaleDatas[i].min);
				}
			}
			return (scaleDatas);
		}

		void SaveScaleDatas(std::vector<ScaleData> scaleDatas, std::string fileName)
		{
			std::ofstream file;
			file.open(fileName);
			for (int i = 0; i < scaleDatas.size(); ++i)
			{
				file << scaleDatas[i].min << " " << scaleDatas[i].max << " " << scaleDatas[i].mean << std::endl;
			}
			file.close();
		}

		std::vector<ScaleData>	LoadScaleDatas(std::string fileName)
		{
			std::vector<ScaleData>		scaleDatas;
			int										nbLine = 0;

			std::string	line;
			std::ifstream file(fileName);
			while (file.eof() == false)
			{
				std::getline(file, line);
				if (line.length() > 0)
				{
					ScaleData	scaleData;
					std::string				element;
					std::istringstream 		elements(line);

					std::getline(elements, element, ' ');
					scaleData.min = std::stod(element);
					std::getline(elements, element, ' ');
					scaleData.max = std::stod(element);
					std::getline(elements, element, ' ');
					scaleData.mean = std::stod(element);

					++nbLine;
					
					scaleDatas.push_back(scaleData);
				}
			}
			file.close();
			return (scaleDatas);
		}
	
	}
}