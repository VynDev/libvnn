/*
* @Author: Vyn
* @Date:   2019-03-12 12:11:21
* @Last Modified by:   Vyn
* @Last Modified time: 2019-04-30 16:28:50
*/

#ifndef CSV_H
#define CSV_H

#include <vector>
#include <iterator>
#include <memory>
#include <iostream>

namespace File
{

	typedef std::vector<std::vector<std::string>> CsvData;
	typedef std::vector<std::vector<double>> CsvNumericData;

	class Csv
	{

	private:

		CsvData data;

	public:

		Csv() {};
		Csv(const Csv &other);
		Csv(Csv &&other);
		Csv(std::string path, bool hasHeader = true);
		Csv(CsvNumericData &NumericData);

		Csv &operator=(Csv &&other);
		Csv &operator=(const Csv &other);
		std::vector<std::string> &operator[](unsigned int i);


		void AddRow(std::vector<std::string> &newRow);
		void DeleteRow(unsigned int index);
		void SetData(CsvData newData) {data = newData;};
		void RemoveLinesWithMissingData();
		void Save(std::string path);

		Csv CopyColumns(std::vector<int> &indexes);
		Csv	CopyRows(int index);
		Csv	CopyRows(int index, int count);
		Csv Merge(Csv &otherCsv);

		CsvNumericData ToNumericVector();
		const CsvData &GetData() const {return (data);};
		const unsigned int GetSize() const {return (data.size());};
	};
}

#endif