#include "CRFModel.h"
#include "Constant.h"

#include <random>
#include <ctime>

using namespace std;

void CRFModel::MHTrain1(Config *conf)
{
	int max_iter         = conf->max_iter;
	int batch_size       = conf->batch_size;
	int max_infer_iter   = conf->max_infer_iter;
	double learning_rate = conf->gradient_step;
	DataSample *sample = train_data->sample[0];
	FactorGraph *graph = &(sample_factor_graph[0]);

	vector<int> state1, state2;
	if (state.size() != N)
	{
		state1.assign(N, 0);
		state2.assign(N, 0);

		for (int i = 0; i < N; i++)
		{
			double maxsum = 0;
			for (int y = 0; y < num_label; y++)
			{
				double sum = 0;
				for (int j = 0; j < sample->node[i]->num_attrib; j++)
				{
					sum += lambda[GetAttribParameterId(y, sample->node[i]->attrib[j])] * sample->node[i]->value[j];
				}
				if (sum > maxsum || y == 0)
				{
					maxsum = sum;
					state1[i] = y;
					state2[i] = y;
				}
			}
			if (sample->node[i]->label_type == Enum::KNOWN_LABEL)
			{
				state1[i] = sample->node[i]->label;
				state2[i] = sample->node[i]->label;
			}
		}
	}
	else
	{
		state1 = state;
		state2 = state;
	}

	vector<double> sum1, sum2;
	sum1.assign(num_feature, 0);
	sum2.assign(num_feature, 0);

	double best_valid_auc = 0;
	int valid_wait = 0;
	double *best_lambda = new double[num_feature];
	memcpy(best_lambda, lambda, num_feature * sizeof(double));

	int update = 0;

	for (int iter = 0; iter < max_iter; iter++)
	{
		if (iter % conf->eval_interval == 0)
		{
			printf("[Iter %d]", iter);
			state = state1;
			double auc = MHEvaluate();
			if (auc > best_valid_auc)
			{
				memcpy(best_lambda, lambda, num_feature * sizeof(double));
				best_valid_auc = auc;
				valid_wait = 0;
			}
			else
			{
				valid_wait++;
				if (valid_wait > conf->early_stop_patience)
				{
					break;
				}
			}
		}

		random_device rd;
		default_random_engine gen(time(0));
		uniform_int_distribution<int> rand_N(0, N - 1);
		uniform_int_distribution<int> rand_CLASS(0, num_label - 1);
		uniform_real_distribution<double> rand_P(0, 1);

		int iters = batch_size;

		map<int,double> gradient1, gradient2;

		for (int it = 0; it < iters; it++)
		{
			int center = rand_N(gen);
			int y = rand_CLASS(gen);
			double p = rand_P(gen);
			bool b1 = train1_sample(1, center, y, p, state1, gradient1);
			bool b2 = train1_sample(2, center, y, p, state2, gradient2);

			if (b1 || b2) {
				map<int,double>::iterator itt;
				for (itt = gradient1.begin(); itt != gradient1.end(); itt++)
				{
					int pid = itt->first;
					double val = itt->second;
					sum1[pid] += val;
				} 
				for (itt = gradient2.begin(); itt != gradient2.end(); itt++)
				{
					int pid = itt->first;
					double val = itt->second;
					sum2[pid] += val;
				}
			}
		}
		double norm = 0;
		norm = double(batch_size);
		for (int pid = 0; pid < num_feature; pid++)
		{
			double val = (sum1[pid] - sum2[pid]);
			sum1[pid] = sum2[pid] = 0;
			lambda[pid] += learning_rate * val / norm;
		}
	}

	memcpy(lambda, best_lambda, num_feature * sizeof(double));
	state = state1;
	MHEvaluate(max_infer_iter, true);
}

bool CRFModel::train1_sample(int type, int center, int ynew, double p, vector<int>& _state, map<int,double>& _gradient)
{
	DataSample *sample = train_data->sample[0];
	FactorGraph *graph = &(sample_factor_graph[0]);

	int y_center;
	double likeli1 = 0, likeli = 0;
	int u = center;
	y_center = _state[center];
	map<int,double> temp1, temp2;
	temp1.clear(); temp2.clear();

	// calculate for Y
	for (int j = 0; j < sample->node[u]->num_attrib; j++)
	{
		int pid = GetAttribParameterId(_state[u], sample->node[u]->attrib[j]);
		double val = sample->node[u]->value[j];
		likeli += lambda[pid] * val;
		temp1[pid] = temp1[pid] + val;
	}

	Node *U = &(graph->var_node[u]);
	for (int j = 0; j < U->neighbor.size(); j++)
	{
		Node *W = U->neighbor[j];
		int w = W->id - N;
		Node *V = NULL;
		for (int k = 0; k < W->neighbor.size(); k++)
		{
			if (W->neighbor[k] != U)
			{
				V = W->neighbor[k];
				break;
			}
		}
		int v = V->id;
		int pid = GetEdgeParameterId(sample->edge[w]->edge_type, _state[u], _state[v]);
		likeli += lambda[pid];
		temp1[pid] = temp1[pid] + 0.5;
	}

	if (type == 1 && sample->node[center]->label_type == Enum::KNOWN_LABEL)
	{
		_gradient = temp1;
		return false;
	}

    // change Y to Ynew
    _state[center] = ynew;

	// calculate for Ynew
	for (int j = 0; j < sample->node[u]->num_attrib; j++)
	{
		int pid = GetAttribParameterId(_state[u], sample->node[u]->attrib[j]);
		double val = sample->node[u]->value[j];
		likeli1 += lambda[pid] * val;
		temp2[pid] = temp2[pid] + val;
	}

	U = &(graph->var_node[u]);
	for (int j = 0; j < U->neighbor.size(); j++)
	{
		Node *W = U->neighbor[j];
		int w = W->id - N;
		Node *V = NULL;
		for (int k = 0; k < W->neighbor.size(); k++)
		{
			if (W->neighbor[k] != U)
			{
				V = W->neighbor[k];
				break;
			}
		}
		int v = V->id;
		int pid = GetEdgeParameterId(sample->edge[w]->edge_type, _state[u], _state[v]);
		likeli1 += lambda[pid];
		temp2[pid] = temp2[pid] + 0.5;
	}

	double acc = min(1.0, exp(likeli1 - likeli));
	if (likeli1 - likeli > 0)
	{
		acc = 1;
	}
	else if (likeli1 - likeli < -64)
	{
		acc = 0;
	}
	if (p > acc) // reject
	{
		_state[center] = y_center;
		_gradient = temp1;
		return false;
	}
	else // accept
	{
		_gradient = temp2;
		return true;
	}
}