#pragma once

#include <string>

using std::string;

#define LBFGS 0
#define GradientDescend 1

class Config
{
public:
    int         my_rank;
    int         num_procs;

    string      task;
	string      method;

    string      train_file;
    string      test_file;
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
    int         optimization_method;

    //bool        eval_each_iter;

    double      penalty_sigma_square;

	int num_thread;
    int batch_size;
	int early_stop_patience;
	string state_file;

    Config(){ SetDefault(); }
    void SetDefault();

    // false => parameter wrong
    bool LoadConfig(int my_rank, int num_procs, int argc, char* argv[]);
    static void ShowUsage();
};