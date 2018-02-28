#pragma once

#include <string>

using std::string;

class Config
{
public:
	string      task;
	string      method;

	string      train_file;
	string      pred_file;

	string      dict_file;
	string      src_model_file;
	string      dst_model_file;

	double      eps;

	int         max_iter;
	int         max_infer_iter;
	int         eval_interval;

	double      gradient_step;

	bool        has_attrib_value;

	int         num_thread;
	int         batch_size;
	int         early_stop_patience;
	string      state_file;

	Config()
	{
		SetDefault();
	}

	void SetDefault();

	// false => parameter wrong
	bool LoadConfig(int argc, char* argv[]);
	static void ShowUsage();
};