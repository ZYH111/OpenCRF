#include <random>
#include <ctime>

#include "CRFModel.h"
#include "Constant.h"

using namespace std;

#define MAX_BUF_SIZE 65536

void CRFModel::LoadStateFile(const char* filename)
{
	FILE* fin = fopen(filename, "r");
    char buf[MAX_BUF_SIZE];
	vector<string> tokens;
	DataSample *sample = train_data->sample[0];
	MappingDict *dict = &(train_data->label_dict);

	state.clear();
	state.assign(N, 0);
	for (int i = 0; i < N; i++)
	{
		if (fgets(buf, MAX_BUF_SIZE, fin) == NULL)
            break;
		if (sample->node[i]->label_type == Enum::KNOWN_LABEL)
		{
			state[i] = sample->node[i]->label;
			continue;
		}
		tokens = CommonUtil::StringTokenize(buf);
        string max_c;
        double max_p = 0;
        for (int j = 0; j < num_label; j++) {
            double p = atof(tokens[2 * j + 1].c_str());
            if (p > max_p) {
                max_p = p;
                max_c = tokens[2 * j];
            }
        }
		state[i] = dict->GetIdConst(max_c);
	}
	fclose(fin);
    printf("Load %s finished.\n", filename);
}

void CRFModel::MHTrain(Config *conf)
{
	int max_iter         = conf->max_iter; 
    int batch_size       = conf->batch_size;
    int max_infer_iter   = conf->max_infer_iter; 
    double learning_rate = conf->gradient_step;
	DataSample *sample = train_data->sample[0];
	FactorGraph *graph = &(sample_factor_graph[0]);

	if (state.size() != N)
	{
		state.clear();
        state.assign(N, 0);

        for (int i = 0; i < N; i++) {
            double maxsum = 0;
            for (int y = 0; y < num_label; y++) {
                double sum = 0;
                for (int j = 0; j < sample->node[i]->num_attrib; j++) {
					sum += lambda[GetAttribParameterId(y, sample->node[i]->attrib[j])] * sample->node[i]->value[j];
                    //sum += lambda[get_node_param_id(y, feature_id[i][j])] * feature_val[i][j];
                }
                if (sum > maxsum || y == 0) {
                    maxsum = sum;
                    state[i] = y;
                }
            }
            if (sample->node[i]->label_type == Enum::KNOWN_LABEL) {
                state[i] = sample->node[i]->label;
            }
        }
	}

	map<int,double> gradient_thread;
	gradient_thread.clear();

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
				if (valid_wait > conf->early_stop_patience) {
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

		for (int batch_iter = 0; batch_iter < iters; batch_iter++) {
            map<int,int> change;
            map<int,double> gradient;

            change.clear(); 
            gradient.clear();
            int auc1 = 0, auc = 0;
            double likeli1 = 0, likeli = 0;
            
            // generate change set
            int center = rand_N(gen);
            change[center] = state[center];
            // for (int i = 0; i < graph->outlist[center].size(); i++) {
            //  int u = graph->outlist[center][i];
            //  change[u] = _state[u];
            // }
            // for (int i = 0; i < graph->inlist[center].size(); i++) {
            //  int u = graph->inlist[center][i];
            //  change[u] = _state[u];
            // }
            // calculate for Y
            map<int,int>::iterator it;
            for (it = change.begin(); it != change.end(); it++) {
                int u = it->first;
                if (sample->node[u]->label_type == Enum::KNOWN_LABEL && state[u] == sample->node[u]->label)
                    auc++;
                for (int j = 0; j < sample->node[u]->num_attrib; j++) {
					int pid = GetAttribParameterId(state[u], sample->node[u]->attrib[j]);
					double val = sample->node[u]->value[j];
                    //int pid = get_node_param_id(_state[u], feature_id[u][j]);
                    //double val = feature_val[u][j];
                    likeli += lambda[pid] * val;
                    gradient[pid] = gradient[pid] - val;
                }
            }
            for (it = change.begin(); it != change.end(); it++) {
                int u = it->first;
				Node *U = &(graph->var_node[u]);

                for (int j = 0; j < U->neighbor.size(); j++) {
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
                    int pid = GetEdgeParameterId(sample->edge[w]->edge_type, state[u], state[v]);
                    likeli += lambda[pid];
                    gradient[pid] = gradient[pid] - 1;
                }
                /*for (int j = 0; j < graph->outlist[u].size(); j++) {
                    int v = graph->outlist[u][j];
                    int pid = get_edge_param_id(_state[u], _state[v]);
                    likeli += lambda[pid];
                    gradient[pid] = gradient[pid] - 1;
                    }
				for (int j = 0; j < graph->inlist[u].size(); j++) {
                    int v = graph->inlist[u][j];
                    if (change.count(v) > 0) continue;
                    int pid = get_edge_param_id(_state[v], _state[u]);
                    likeli += lambda[pid];
                    gradient[pid] = gradient[pid] - 1;
                }*/
            }
            // change Y to Ynew
            for (it = change.begin(); it != change.end(); it++) {
                int u = it->first;
                state[u] = rand_CLASS(gen); //sample(u, _state); //
            }
            // calculate for Ynew
            for (it = change.begin(); it != change.end(); it++) {
                int u = it->first;
                //if (label_type[u] == KNOWN_LABEL && _state[u] == label[u])
				if (sample->node[u]->label_type == Enum::KNOWN_LABEL && state[u] == sample->node[u]->label)
                    auc1++;
                //for (int j = 0; j < feature_id[u].size(); j++) {
				for (int j = 0; j < sample->node[u]->num_attrib; j++) {
					int pid = GetAttribParameterId(state[u], sample->node[u]->attrib[j]);
					double val = sample->node[u]->value[j];
                    //int pid = get_node_param_id(_state[u], feature_id[u][j]);
                    //double val = feature_val[u][j];
                    likeli1 += lambda[pid] * val;
                    gradient[pid] = gradient[pid] + val;
                }
            }
            for (it = change.begin(); it != change.end(); it++) {
                int u = it->first;
				Node *U = &(graph->var_node[u]);

				for (int j = 0; j < U->neighbor.size(); j++) {
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
                    int pid = GetEdgeParameterId(sample->edge[w]->edge_type, state[u], state[v]);
                    likeli1 += lambda[pid];
                    gradient[pid] = gradient[pid] + 1;
                }
                /*for (int j = 0; j < graph->outlist[u].size(); j++) {
                    int v = graph->outlist[u][j];
                    int pid = get_edge_param_id(_state[u], _state[v]);
                    likeli1 += lambda[pid];
                    gradient[pid] = gradient[pid] + 1;
                }
                for (int j = 0; j < graph->inlist[u].size(); j++) {
                    int v = graph->inlist[u][j];
                    if (change.count(v) > 0) continue;
                    int pid = get_edge_param_id(_state[v], _state[u]);
                    likeli1 += lambda[pid];
                    gradient[pid] = gradient[pid] + 1;
                }*/
            }
            // accept/reject Ynew
            //printf("%f, %f  %d, %d\n", likeli1, likeli, auc1, auc);
            double acc = min(1.0, exp(likeli1 - likeli));
            if (likeli1 - likeli > 0) {
                acc = 1;
            } else if (likeli1 - likeli < -64) {
                acc = 0;
            }
            double p = rand_P(gen);
            if (p > acc) { // reject
                for (it = change.begin(); it != change.end(); it++) {
                    int u = it->first;
                    state[u] = it->second;
                }
            } else {
                // update lambda
                double step = 0;
                if (auc1 > auc && likeli1 <= likeli) {
                    step = learning_rate;
                } else if (auc1 < auc && likeli1 > likeli) {
                    step = -learning_rate;
                }
                if (step != 0) {
                    update += 1;
                    double norm = double(batch_size);
                    map<int,double>::iterator itt;
                    //norm = 1;
                    for (itt = gradient.begin(); itt != gradient.end(); itt++) {
                        int id = itt->first;
                        double val = itt->second / norm;
                        gradient_thread[id] = gradient_thread[id] + step * val;
                    }
                }
            }
        }
		map<int,double>::iterator itt;
        for (itt = gradient_thread.begin(); itt != gradient_thread.end(); itt ++) {
            int id = itt->first;
            double val = itt->second;
            lambda[id] += val;
        }
	}

	//lambda = best_lambda;
	memcpy(lambda, best_lambda, num_feature * sizeof(double));
    MHEvaluate(max_infer_iter);
}

double CRFModel::MHEvaluate(int max_iter)
{
	DataSample *sample = train_data->sample[0];
	FactorGraph *graph = &(sample_factor_graph[0]);

	unlabeled.clear();
	test.clear();
	valid.clear();
	for (int i = 0; i < N; i++)
		if (sample->node[i]->label_type == Enum::UNKNOWN_LABEL)
		{
			unlabeled.push_back(i);
			if (sample->node[i]->type1 == Enum1::VALID) valid.push_back(i);
			else test.push_back(i);
		}

	if (state.size() != N)
	{
		state.clear();
        state.assign(N, 0);

        for (int i = 0; i < N; i++) {
            double maxsum = 0;
            for (int y = 0; y < num_label; y++) {
                double sum = 0;
                for (int j = 0; j < sample->node[i]->num_attrib; j++) {
					sum += lambda[GetAttribParameterId(y, sample->node[i]->attrib[j])] * sample->node[i]->value[j];
                    //sum += lambda[get_node_param_id(y, feature_id[i][j])] * feature_val[i][j];
                }
                if (sum > maxsum || y == 0) {
                    maxsum = sum;
                    state[i] = y;
                }
            }
        }
	}

	for (int i = 0; i < N; i++)
		if (sample->node[i]->label_type == Enum::KNOWN_LABEL) {
                state[i] = sample->node[i]->label;
        }

	best_state = state;
    printf("EVAL#"); 
    fflush(stdout);
	double best_likeli = 0;

	std::random_device rd;
    std::default_random_engine gen(time(0));
    std::uniform_int_distribution<int> rand_U(0, unlabeled.size() - 1);
    std::uniform_int_distribution<int> rand_CLASS(0, num_label - 1);
    std::uniform_real_distribution<double> rand_P(0, 1);

	map<int,int> change;
    double state_likeli = 0;
    int iters = max_iter;
    for (int iter = 0; iter < iters; iter++) {
        // if (iter % (max_iter / 10) == 0) {
            // printf("."); fflush(stdout);
        // }
        change.clear(); 
        int auc1 = 0, auc = 0;
        double likeli1 = 0, likeli = 0;
        // generate change set
        int center = unlabeled[rand_U(gen)];
        if (sample->node[center]->label_type == Enum::KNOWN_LABEL) {
            continue;
        }
        change[center] = state[center];
        // calculate for Y
        map<int,int>::iterator it;
        for (it = change.begin(); it != change.end(); it++) {
            int u = it->first;
			for (int j = 0; j < sample->node[u]->num_attrib; j++) {
            //for (int j = 0; j < feature_id[u].size(); j++) {
                //int pid = get_node_param_id(_state[u], feature_id[u][j]);
                //double val = feature_val[u][j];
				int pid = GetAttribParameterId(state[u], sample->node[u]->attrib[j]);
				double val = sample->node[u]->value[j];
                likeli += lambda[pid] * val;
            }
        }
        for (it = change.begin(); it != change.end(); it++) {
            int u = it->first;
			Node *U = &(graph->var_node[u]);

            for (int j = 0; j < U->neighbor.size(); j++) {
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
                int pid = GetEdgeParameterId(sample->edge[w]->edge_type, state[u], state[v]);
				likeli += lambda[pid];
			}
			
            /*for (int j = 0; j < graph->outlist[u].size(); j++) {
                int v = graph->outlist[u][j];
                int pid = get_edge_param_id(_state[u], _state[v]);
                likeli += lambda[pid];
            }
            for (int j = 0; j < graph->inlist[u].size(); j++) {
                int v = graph->inlist[u][j];
                if (change.count(v) > 0) continue;
                int pid = get_edge_param_id(_state[v], _state[u]);
                likeli += lambda[pid];
            }*/
        }
        // change Y to Ynew
        for (it = change.begin(); it != change.end(); it++) {
            int u = it->first;
            state[u] = rand_CLASS(gen);
        }
        // calculate for Ynew
        for (it = change.begin(); it != change.end(); it++) {
            int u = it->first;
			for (int j = 0; j < sample->node[u]->num_attrib; j++) {
            //for (int j = 0; j < feature_id[u].size(); j++) {
                //int pid = get_node_param_id(_state[u], feature_id[u][j]);
                //double val = feature_val[u][j];
				int pid = GetAttribParameterId(state[u], sample->node[u]->attrib[j]);
				double val = sample->node[u]->value[j];
                likeli1 += lambda[pid] * val;
            }
        }
        for (it = change.begin(); it != change.end(); it++) {
            int u = it->first;
			Node *U = &(graph->var_node[u]);

			for (int j = 0; j < U->neighbor.size(); j++) {
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
                int pid = GetEdgeParameterId(sample->edge[w]->edge_type, state[u], state[v]);
                likeli1 += lambda[pid];
			}
            /*for (int j = 0; j < graph->outlist[u].size(); j++) {
                int v = graph->outlist[u][j];
                int pid = get_edge_param_id(_state[u], _state[v]);
                likeli1 += lambda[pid];
            }
            for (int j = 0; j < graph->inlist[u].size(); j++) {
                int v = graph->inlist[u][j];
                if (change.count(v) > 0) continue;
                int pid = get_edge_param_id(_state[v], _state[u]);
                likeli1 += lambda[pid];
            }*/
        }
        // accept/reject Ynew
        double acc = min(1.0, exp(likeli1 - likeli));
        if (likeli1 - likeli > 0) {
            acc = 1;
        } else if (likeli1 - likeli < -64) {
            acc = 0;
        }
        double p = rand_P(gen);
        if (p > acc) { // reject
            for (it = change.begin(); it != change.end(); it++) {
                int u = it->first;
                state[u] = it->second;
            }
        } else {
            state_likeli = state_likeli + likeli1 - likeli;
        }
        if (state_likeli > best_likeli) {
            //printf("update\n");
            best_state = state;
            best_likeli = state_likeli;
        }
    }

	vector<int> tmp_state = best_state;
	printf("."); fflush(stdout);
    for (int i = 0; i < unlabeled.size(); i++) {
        int u = unlabeled[i];
        vector<double> p;
        p.clear();
        p.assign(num_label, 0);
        for (int k = 0; k < num_label; k++) {
			for (int j = 0; j < sample->node[u]->num_attrib; j++) {	
            //for (int j = 0; j < feature_id[u].size(); j++) {
				int pid = GetAttribParameterId(k, sample->node[u]->attrib[j]);
				double val = sample->node[u]->value[j];
                //int pid = get_node_param_id(k, feature_id[u][j]);
                //double val = feature_val[u][j];
                p[k] += lambda[pid] * val;
            }
			Node *U = &(graph->var_node[u]);

            for (int j = 0; j < U->neighbor.size(); j++) {
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
                int pid = GetEdgeParameterId(sample->edge[w]->edge_type, k, best_state[v]);
				p[k] += lambda[pid];
			}
            /*for (int j = 0; j < graph->outlist[u].size(); j++) {
                int v = graph->outlist[u][j];
                int pid = get_edge_param_id(k, best_state[v]);
                p[k] += lambda[pid];
            }
            for (int j = 0; j < graph->inlist[u].size(); j++) {
                int v = graph->inlist[u][j];
                int pid = get_edge_param_id(best_state[v], k);
                p[k] += lambda[pid];
            }*/
        }
        int ynew = 0;
        for (int k = 0; k < num_label; k++) {
            if (p[k] > p[ynew]) {
                ynew = k;
            }
        }
        if (ynew != best_state[u]) {
            tmp_state[u] = ynew;
        }
    }

	int hit = 0, all = 0;
    for (int i = 0; i < test.size(); i++) {
        int u =  test[i];
        all += 1;
        hit += (sample->node[u]->label == tmp_state[u]);
    }
    int vhit = 0, vall = 0;
    for (int i = 0; i < valid.size(); i++) {
        int u = valid[i];
        vall += 1;
        vhit += (sample->node[u]->label == tmp_state[u]);   
    }
    double valid_auc = double(vhit) / double(max(vall, 1));
    printf("Accuracy: %d / %d = %.4f, Valid: %.4f\n", hit, all, double(hit) / double(all), valid_auc);
    fflush(stdout);
    //state = origin_state;
    return valid_auc;
}

void CRFModel::SavePred(const char* filename)
{
	FILE* fout = fopen(filename, "w");
	DataSample *sample = train_data->sample[0];
	FactorGraph *graph = &(sample_factor_graph[0]);
	MappingDict *dict = &(train_data->label_dict);

    // for (int i = 0; i < N; i++) {
    //    fprintf(fout, "%d\n", best_state[i]);
    // }
    vector<double> p;
    for (int u = 0; u < N; u++) {
        p.clear();
        p.assign(num_label, 0);
        for (int k = 0; k < num_label; k++) {
			for (int j = 0; j < sample->node[u]->num_attrib; j++) {	
            //for (int j = 0; j < feature_id[u].size(); j++) {
				int pid = GetAttribParameterId(k, sample->node[u]->attrib[j]);
				double val = sample->node[u]->value[j];
                //int pid = get_node_param_id(k, feature_id[u][j]);
                //double val = feature_val[u][j];
                p[k] += lambda[pid] * val;
            }
			Node *U = &(graph->var_node[u]);

            for (int j = 0; j < U->neighbor.size(); j++) {
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
                int pid = GetEdgeParameterId(sample->edge[w]->edge_type, k, best_state[v]);
				p[k] += lambda[pid];
			}
            /*for (int j = 0; j < feature_id[i].size(); j++) {
                int pid = get_node_param_id(k, feature_id[i][j]);
                double val = feature_val[i][j];
                p[k] += lambda[pid] * val;
            }
            for (int j = 0; j < graph->outlist[i].size(); j++) {
                int v = graph->outlist[i][j];
                int pid = get_edge_param_id(k, best_state[v]);
                p[k] += lambda[pid];
            }
            for (int j = 0; j < graph->inlist[i].size(); j++) {
                int v = graph->inlist[i][j];
                int pid = get_edge_param_id(best_state[v], k);
                p[k] += lambda[pid];
            }*/
        }
        double pmin = p[0], pmax = p[0];
        for (int k = 0; k < num_label; k++) {
            pmin = min(pmin, p[k]);
            pmax = max(pmax, p[k]);
        }
        for (int k = 0; k < num_label; k++) {
            // p[k] = (p[k] - pmin) / (pmax - pmin);
            fprintf(fout, "%s %.3f ", dict->GetKeyWithId(k).c_str(), p[k]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
}