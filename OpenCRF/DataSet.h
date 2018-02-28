#pragma once

#include "Util.h"
#include "Config.h"

#include <string>
#include <vector>
#include <map>
using std::string;
using std::vector;
using std::map;

class DataNode
{
public:
	int                 label_type;
	int                 type1;
	int                 label;
	int                 num_attrib;
	vector<int>         attrib;
	vector<double>      value;

	int GetSize() const
	{
		return sizeof(int) * (3 + num_attrib) + sizeof(double) * num_attrib;
	}
};

class DataEdge
{
public:
	int                 a, b, edge_type;

	int GetSize() const
	{
		return sizeof(int) * 3;
	}
};

class DataSample
{
public:
	int                 num_node;
	int                 num_edge;
	vector<DataNode*>   node;
	vector<DataEdge*>   edge;

	~DataSample()
	{
		for (int i = 0; i < node.size(); i++)
			delete node[i];
		for (int i = 0; i < edge.size(); i++)
			delete edge[i];
	}

	int GetSize() const
	{
		int size = sizeof(int) * 2;
		for (int i = 0; i < node.size(); i++)
			size += node[i]->GetSize();
		for (int i = 0; i < edge.size(); i++)
			size += edge[i]->GetSize();
		return size;
	}
};

class DataSet
{
public:
	vector<DataSample*> sample;

	int num_label;
	int num_sample;
	int num_attrib_type;
	int num_edge_type;

	MappingDict         label_dict;
	MappingDict         attrib_dict;
	MappingDict         edge_type_dict;

	void LoadData(const char* data_file, Config* conf);
	void LoadDataWithDict(const char* data_file, Config* conf, const MappingDict& ref_label_dict, const MappingDict& ref_attrib_dict, const MappingDict& ref_edge_type_dict);

	~DataSet()
	{
		for (int i = 0; i < sample.size(); i++)
			delete sample[i];
	}
};