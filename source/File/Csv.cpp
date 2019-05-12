/*
* @Author: Vyn
* @Date:   2019-03-12 12:11:21
* @Last Modified by:   Vyn
* @Last Modified time: 2019-05-12 13:35:49
*/

#include <fstream>
#include <sstream>
#include <iostream>

#include "Csv.h"

namespace File
{
	Csv::Csv(std::string path, bool hasHeader)
	{
		int nbLine = 0;
		int nbColumn = -1;
		int tmpNbColumn;

		std::ifstream file(path);
		while (file.eof() == false)
		{
			++nbLine;
			std::string	line;
			std::getline(file, line);

			if (line.length() > 0)
			{
				for (int i = 0; i < line.length(); ++i)
				{
					if (line[i] == '"')
					{
						line[i] = 'A';
						while (i < line.length() && line[i] != '"')
						{
							line[i] = 'A';
							++i;
						}
						if (i < line.length())
							line[i] = 'B';
					}
				}
				std::vector<std::string> csvLine;
				std::istringstream elements(line);

				tmpNbColumn = 0;
				while (elements.eof() == false)
				{
					++tmpNbColumn;
					std::string	element;
					std::getline(elements, element, ',');
					if (element[element.length() - 1] == '\r')
						element.erase(element.begin() + element.length() - 1);
					csvLine.push_back(element);
				}
				if (nbColumn == -1)
					nbColumn = tmpNbColumn;
				else if (nbColumn != tmpNbColumn)
					throw std::string("CSV: Not the same number of columns");
				data.push_back(csvLine);
			}
		}
		if (hasHeader && data.size() > 0)
			data.erase(data.begin());
		file.close();
	}

	Csv::Csv(const Csv &other)
	{
		//std::cout << "copy constructor" << std::endl;
		this->SetData(other.GetData());
	}

	Csv::Csv(Csv &&other)
	{
		//std::cout << "move constructor" << std::endl;
		this->SetData(other.GetData());
	}

	Csv::Csv(CsvNumericData &NumericData)
	{
		for (int i = 0; i < NumericData.size(); ++i)
		{	
			std::vector<std::string> line;
			for (int j = 0; j < NumericData[i].size(); ++j)
			{
				line.push_back(std::to_string(NumericData[i][j]));
			}
			data.push_back(line);
		}
	}

	Csv &Csv::operator=(Csv &&other)
	{
		//std::cout << "move" << std::endl;
		this->SetData(other.GetData());
		return *this;
	}

	Csv &Csv::operator=(const Csv &other)
	{
		//std::cout << "copy" << std::endl;
		this->SetData(other.GetData());
		return *this;
	}

	std::vector<std::string> &Csv::operator[](unsigned int i)
	{
		return (data[i]);
	}

	void Csv::Save(std::string path)
	{
		std::ofstream file;
		file.open(path);
		for (unsigned int i = 0; i < data.size(); ++i)
		{
			for (unsigned int j = 0; j < data[i].size(); ++j)
			{
				file << data[i][j];
				if (j != data[i].size() - 1)
					file << ",";
			}
			file << std::endl;
		}
		file.close();
	}

	void Csv::AddRow(std::vector<std::string> &newRow)
	{
		data.push_back(newRow);
	}

	void Csv::DeleteRow(unsigned int index)
	{
		data.erase(data.begin() + index);
	}

	CsvNumericData Csv::ToNumericVector()
	{
		CsvNumericData numericCsvData;
		for (int i = 0; i < data.size(); ++i)
		{
			std::vector<double> line;
			for (int j = 0; j < data[0].size(); ++j)
			{
				line.push_back(std::stod(data[i][j]));
			}
			numericCsvData.push_back(line);
		}
		return (numericCsvData);
	}

	Csv Csv::CopyColumns(std::vector<int> &indexes)
	{
		Csv newCsv;

		for (int rowIndex = 0; rowIndex < data.size(); ++rowIndex)
		{
			std::vector<std::string> line;
			for (int i = 0; i < indexes.size(); ++i)
			{
				line.push_back(data[rowIndex][indexes[i]]);
			}
			newCsv.AddRow(line);
		}
		return (newCsv);
	}

	Csv	Csv::CopyRows(int index)
	{
		Csv newCsv;

		if (index >= 0)
		{
			for (int i = index; i < data.size(); ++i)
			{
				newCsv.AddRow(data[i]);
			}
		}
		else if (index < 0)
		{
			for (int i = data.size() + index; i < data.size(); ++i)
			{
				newCsv.AddRow(data[i]);
			}
		}
		return (newCsv);
	}

	Csv	Csv::CopyRows(int index, int count)
	{
		Csv newCsv;

		for (int i = 0; index + i < data.size() && i < count; ++i)
		{
			newCsv.AddRow(data[index + i]);
		}
		return (newCsv);
	}

	Csv Csv::Merge(Csv &otherCsv)
	{
		Csv	newCsv;
		for (unsigned int i = 0; i < data.size(); ++i)
		{
			std::vector<std::string> line;
			for (unsigned int j = 0; j < data[i].size(); ++j)
			{
				line.push_back(data[i][j]);
			}
			for (unsigned int j = 0; j < otherCsv[i].size(); ++j)
			{
				line.push_back(otherCsv[i][j]);
			}
			newCsv.AddRow(line);
		}
		return (newCsv);
	}

	void Csv::RemoveLinesWithMissingData()
	{
		for (int i = 0; i < data.size(); ++i)
		{
			bool shouldErase = false;
			for (int j = 0; j < data[0].size(); ++j)
			{
				if (data[i][j].length() == 0)
					shouldErase = true;
			}
			if (shouldErase)
			{
				data.erase(data.begin() + i);
				--i;
			}
		}
	}
}