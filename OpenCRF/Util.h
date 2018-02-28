#pragma once

#include <string>
#include <vector>
#include <map>

using std::string;
using std::vector;
using std::map;

class MappingDict
{
public:
	map<string, int>    dict;
	vector<string>      keys;
	
	int GetSize() const
	{
		return keys.size();
	}
	
	int GetId(const string &key);            // insert if not exist
	int GetIdConst(const string &key) const; // return -1 (if not exist)

	string GetKeyWithId(const int id) const; // return "" (if not exist)

	void SaveMappingDict(const char* file_name);
	void LoadMappingDict(const char* file_name);

	void clear()
	{
		dict.clear();
		keys.clear();
	}
};

class CommonUtil
{
public:
	static vector<string> StringTokenize(string line);
	static vector<string> StringSplit(string line, char separator);
};

class MatrixUtil
{
public:
	static double* GetDoubleArr(int size)
	{
		double* arr = new double[size];
		return arr;
	}

	static void DoubleArrFill(double* arr, int size, double v)
	{
		for (int i = 0; i < size; i ++)
			arr[i] = v;
	}
};