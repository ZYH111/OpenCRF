#include "CRFModel.h"
#include "Constant.h"
#include <ctime>

#define MAX_BUF_SIZE 65536

void CRFModel::InitTrain(Config* conf, DataSet* train_data)
{
	this->conf = conf;
	this->train_data = train_data;

	num_sample = train_data->num_sample;
	num_label = train_data->num_label;
	num_attrib_type = train_data->num_attrib_type;
	num_edge_type = train_data->num_edge_type;

	GenFeature();
	lambda = new double[num_feature];
	// Initialize parameters
	for (int i = 0; i < num_feature; i++)
		lambda[i] = 0.0;
	SetupFactorGraphs();

	N = train_data->sample[0]->num_node;
	M = train_data->sample[0]->num_edge;
}

void CRFModel::GenFeature()
{
	num_feature = 0;

	// state feature: f(y, x)
	num_attrib_parameter = num_label * num_attrib_type;
	num_feature += num_attrib_parameter;

	// edge feature: f(edge_type, y1, y2)
	edge_feature_offset.clear();
	int offset = 0;
	for (int y1 = 0; y1 < num_label; y1++)
		for (int y2 = y1; y2 < num_label; y2++)
		{
			edge_feature_offset.insert( make_pair(y1 * num_label + y2, offset) );
			offset ++;
		}
	num_edge_feature_each_type = offset;
	num_feature += num_edge_type * num_edge_feature_each_type;
}

void CRFModel::SetupFactorGraphs()
{
	double* p_lambda = lambda + num_attrib_parameter;
	func_list = new EdgeFactorFunction*[num_edge_type];
	for (int i = 0; i < num_edge_type; i++)
	{
		func_list[i] = new EdgeFactorFunction(num_label, p_lambda, &edge_feature_offset);
		p_lambda += num_edge_feature_each_type;
	}

	sample_factor_graph = new FactorGraph[num_sample];
	for (int s = 0; s < num_sample; s++)
	{
		DataSample* sample = train_data->sample[s];
		
		int n = sample->num_node;
		int m = sample->num_edge;

		sample_factor_graph[s].InitGraph(n, m, num_label);

		// Add node info
		for (int i = 0; i < n; i++)
		{
			sample_factor_graph[s].SetVariableLabel(i, sample->node[i]->label);
			sample_factor_graph[s].var_node[i].label_type = sample->node[i]->label_type;
		}

		// Add edge info
		for (int i = 0; i < m; i++)
		{
			sample_factor_graph[s].AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type]);
		}

		sample_factor_graph[s].GenPropagateOrder();
	}
}

void CRFModel::Train()
{    
	double* gradient;
	double  f;          // log-likelihood

	gradient = new double[num_feature + 1];

	///// Initilize all info

	// Data Varible         
	double  old_f = 0.0;

	// Variable for optimization
	int     m_correlation = 3;
	double* work_space = new double[num_feature * (2 * m_correlation + 1) + 2 * m_correlation];
	int     diagco = 0;
	double* diag = new double[num_feature];
	int     iprint[2] = {-1, 0}; // do not print anything
	double  eps = conf->eps;
	double  xtol = 1.0e-16;
	int     iflag = 0;

	// Other Variables
	int     num_iter;
	double  *tmp_store = new double[num_feature + 1];

	// Main-loop of CRF
	// Paramater estimation via Gradient Descend
	num_iter = 0;

	double start_time, end_time;

	do {
		num_iter++;
		start_time = clock() / (double)CLK_TCK;

		// Step A. Calc gradient and log-likehood of the local datas
		f = CalcGradient(gradient);

		// Step B. Opitmization by Gradient Descend
		printf("[Iter %3d] log-likelihood : %.8lf\n", num_iter, f);
		fflush(stdout);

		// If diff of log-likelihood is small enough, break.
		if (fabs(old_f - f) < eps) break;
		old_f = f;

		// Normalize Graident
		double g_norm = 0.0;
		for (int i = 0; i < num_feature; i++)
			g_norm += gradient[i] * gradient[i];
		g_norm = sqrt(g_norm);
			
		if (g_norm > 1e-8)
		{
			for (int i = 0; i < num_feature; i++)
				gradient[i] /= g_norm;
		}

		for (int i = 0; i < num_feature; i++)
			lambda[i] += gradient[i] * conf->gradient_step;
		iflag = 1;

		if (num_iter % conf->eval_interval == 0)
		{
			SelfEvaluate();
		}

		end_time = clock() / (double)CLK_TCK;

		FILE* ftime = fopen("time.out", "a");
		fprintf(ftime, "start_time = %.6lf\n", start_time);
		fprintf(ftime, "end_time = %.6lf\n", end_time);
		fprintf(ftime, "cost = %.6lf\n", end_time - start_time);
		
		fclose(ftime);

		printf("!!! Time cost = %.6lf\n", end_time - start_time);
		fflush(stdout);

	} while (iflag != 0 && num_iter < conf->max_iter);

	delete[] tmp_store;

	delete[] work_space;
	delete[] diag;

	delete[] gradient;
}

double CRFModel::CalcGradient(double* gradient)
{
	double  f;
	
	// Initialize

	f = 0.0;
	for (int i = 0; i < num_feature; i++)
	{
		gradient[i] = 0;
	}

	// Calculation
	for (int i = 0; i < num_sample; i++)
	{
		double t = CalcPartialLabeledGradientForSample(train_data->sample[i], &sample_factor_graph[i], gradient);
		f += t;		 
	}
	
	return f;
}

double CRFModel::CalcPartialLabeledGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient)
{   
	int n = sample->num_node;
	int m = sample->num_edge;
	
	//****************************************************************
	// Belief Propagation 1: labeled data are given.
	//****************************************************************

	factor_graph->labeled_given = true;
	factor_graph->ClearDataForSumProduct();
	
	// Set state_factor
	for (int i = 0; i < n; i++)
	{
		double* p_lambda = lambda;
		for (int y = 0; y < num_label; y++)
		{
			if (sample->node[i]->label_type == Enum::KNOWN_LABEL && y != sample->node[i]->label)
			{
				factor_graph->SetVariableStateFactor(i, y, 0);
			}
			else
			{
				double v = 1;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);                
				factor_graph->SetVariableStateFactor(i, y, v);
			}
			p_lambda += num_attrib_type;
		}
	}
	
	factor_graph->BeliefPropagation(conf->max_infer_iter);
	factor_graph->CalculateMarginal();    

	/***
	* Gradient = E_{Y|Y_L} f_i - E_{Y} f_i
	*/

	// calc gradient part : + E_{Y|Y_L} f_i
	for (int i = 0; i < n; i++)
	{
		for (int y = 0; y < num_label; y++)
		{
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				gradient[GetAttribParameterId(y, sample->node[i]->attrib[t])] += sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				gradient[GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] += factor_graph->factor_node[i].marginal[a][b];
			}
    }

	//****************************************************************
	// Belief Propagation 2: labeled data are not given.
	//****************************************************************

	factor_graph->ClearDataForSumProduct();
	factor_graph->labeled_given = false;

	for (int i = 0; i < n; i++)
	{
		double* p_lambda = lambda;
		for (int y = 0; y < num_label; y++)
		{
			double v = 1;
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
			factor_graph->SetVariableStateFactor(i, y, v);
			p_lambda += num_attrib_type;
		}
	}    

	factor_graph->BeliefPropagation(conf->max_infer_iter);
	factor_graph->CalculateMarginal();
	
	// calc gradient part : - E_{Y} f_i
	for (int i = 0; i < n; i++)
	{
		for (int y = 0; y < num_label; y++)
		{
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				gradient[GetAttribParameterId(y, sample->node[i]->attrib[t])] -= sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				gradient[GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] -= factor_graph->factor_node[i].marginal[a][b];
			}
	}
	
	// Calculate gradient & log-likelihood
	double f = 0.0, Z = 0.0;

	// \sum \lambda_i * f_i
	for (int i = 0; i < n; i++)
	{
		int y = sample->node[i]->label;
		for (int t = 0; t < sample->node[i]->num_attrib; t++)
			f += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t];
	}
	for (int i = 0; i < m; i++)
	{
		int a = sample->node[sample->edge[i]->a]->label;
		int b = sample->node[sample->edge[i]->b]->label;        
		f += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)];
	}

	// calc log-likelihood
	//  using Bethe Approximation
	for (int i = 0; i < n; i++)
	{
		for (int y = 0; y < num_label; y++)
		{
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				Z += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				Z += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
			}
	}
	// Edge entropy
	for (int i = 0; i < m; i++)
	{
		double h_e = 0.0;
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
					h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
			}
		Z += h_e;
	}
	// Node entroy
	for (int i = 0; i < n; i++)
	{
		double h_v = 0.0;
		for (int a = 0; a < num_label; a++)
			if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
				h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
		Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
	}
	
	f -= Z;
	
	// Let's take a look of current accuracy

	factor_graph->ClearDataForMaxSum();
	factor_graph->labeled_given = true;

	for (int i = 0; i < n; i++)
	{
		double* p_lambda = lambda;

		for (int y = 0; y < num_label; y++)
		{
			double v = 1.0;
			for (int t = 0; t < sample->node[i]->num_attrib; t++)
				v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
			factor_graph->SetVariableStateFactor(i, y, v);

			p_lambda += num_attrib_type;
		}
	}    

	factor_graph->MaxSumPropagation(conf->max_infer_iter);

	int* inf_label = new int[n];
	double** label_prob = new double*[num_label];
	for (int p = 0; p < num_label; p++)
		label_prob[p] = new double[n];

	for (int i = 0; i < n; i++)
	{
		int ybest = -1;
		double vbest, v;
		double vsum = 0.0;
		for (int y = 0; y < num_label; y++)
		{
			v = factor_graph->var_node[i].state_factor[y];
			for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
				v *= factor_graph->var_node[i].belief[t][y];
			if (ybest < 0 || v > vbest)
				ybest = y, vbest = v;

			label_prob[y][i] = v;
			vsum += v;
		}

		inf_label[i] = ybest;

		for (int y = 0; y < num_label; y++)
			label_prob[y][i] /= vsum;
	}

	int hit = 0, miss = 0;
	int hitu = 0, missu = 0;

	int cnt[10][10];
	int ucnt[10][10];

	memset(cnt, 0, sizeof(cnt));
	memset(ucnt, 0, sizeof(ucnt));
	FILE *pred_out = fopen("pred.txt", "w");

	for (int i = 0; i < n; i++)
	{
		fprintf(pred_out, "%s\n", train_data->label_dict.GetKeyWithId(inf_label[i]).c_str());
		if (inf_label[i] == sample->node[i]->label)
			hit++;
		else
			miss++;

		cnt[inf_label[i]][sample->node[i]->label]++;

		if (sample->node[i]->label_type == Enum::UNKNOWN_LABEL)
		{
			if (inf_label[i] == sample->node[i]->label)
				hitu++;
			else
				missu++;

			ucnt[inf_label[i]][sample->node[i]->label]++;
		}
	}
	fclose(pred_out);

	int dat[12];
	memset(dat, 0, sizeof(dat));

	hit += dat[0]; hitu += dat[1];
	miss += dat[2]; missu += dat[3];
	cnt[0][0] += dat[4]; cnt[0][1] += dat[5]; cnt[1][0] += dat[6]; cnt[1][1] += dat[7];
	ucnt[0][0] += dat[8]; ucnt[0][1] += dat[9]; ucnt[1][0] += dat[10]; ucnt[1][1] += dat[11];

	printf("A_HIT  = %4d, U_HIT  = %4d\n", hit, hitu);
	printf("A_MISS = %4d, U_MISS = %4d\n", miss, missu);

	//!!!!!!!! make sure, the first instance is "positive"

	// 0 -> positive
	// 1 -> negative

	double ap = (double)cnt[0][0] / (cnt[0][0] + cnt[0][1]);
	double up = (double)ucnt[0][0] / (ucnt[0][0] + ucnt[0][1]);

	double ar = (double)cnt[0][0] / (cnt[0][0] + cnt[1][0]);
	double ur = (double)ucnt[0][0] / (ucnt[0][0] + ucnt[1][0]);

	double af = 2 * ap * ar / (ap + ar);
	double uf = 2 * up * ur / (up + ur);

	printf("A_Accuracy  = %.4lf     U_Accuracy  = %.4lf\n", (double)hit / (hit + miss), (double)hitu / (hitu + missu));
	printf("A_Precision = %.4lf     U_Precision = %.4lf\n", ap, up);
	printf("A_Recall    = %.4lf     U_Recall    = %.4lf\n", ar, ur);
	printf("A_F1        = %.4lf     U_F1        = %.4lf\n", af, uf);
		
	fflush(stdout);

	FILE* fprob = fopen("uncertainty.txt", "w");
	for (int i = 0; i < n; i++)
	{
		if (sample->node[i]->label_type == Enum::KNOWN_LABEL)
		{
			for (int y = 0; y < num_label; y++)
				fprintf(fprob, "%s -1 ", train_data->label_dict.GetKeyWithId(y).c_str());
			fprintf(fprob, "\n");
		}
		else
		{
			for (int y = 0; y < num_label; y++)
				fprintf(fprob, "%s %.4lf ", train_data->label_dict.GetKeyWithId(y).c_str(), label_prob[y][i]);
			fprintf(fprob, "\n");
		}
	}
	fclose(fprob);
    

	delete[] inf_label;
	for (int y = 0; y < num_label; y++)
		delete[] label_prob[y];
	delete[] label_prob;

	return f;
}

void CRFModel::SelfEvaluate()
{
	int ns = train_data->num_sample;
	int tot, hit;

	tot = hit = 0;
	for (int s = 0; s < ns; s++)
	{
		DataSample* sample = train_data->sample[s];
		FactorGraph* factor_graph = &sample_factor_graph[s];
		
		int n = sample->num_node;
		int m = sample->num_edge;
		
		factor_graph->InitGraph(n, m, num_label);
		// Add edge info
		for (int i = 0; i < m; i++)
		{
			factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type]);
		}        
		factor_graph->GenPropagateOrder();

		factor_graph->ClearDataForMaxSum();

		for (int i = 0; i < n; i++)
		{
			double* p_lambda = lambda;

			for (int y = 0; y < num_label; y++)
			{
				double v = 1.0;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
				factor_graph->SetVariableStateFactor(i, y, v);

				p_lambda += num_attrib_type;
			}
		}    

		factor_graph->MaxSumPropagation(conf->max_infer_iter);

		int* inf_label = new int[n];
		for (int i = 0; i < n; i++)
		{
			int ybest = -1;
			double vbest, v;

			for (int y = 0; y < num_label; y++)
			{
				v = factor_graph->var_node[i].state_factor[y];
				for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
					v *= factor_graph->var_node[i].belief[t][y];
				if (ybest < 0 || v > vbest)
					ybest = y, vbest = v;
			}

			inf_label[i] = ybest;
		}

		int curt_tot, curt_hit;
		curt_tot = curt_hit = 0;
		for (int i = 0; i < n; i++)
		{   
			curt_tot ++;
			if (inf_label[i] == sample->node[i]->label) curt_hit++;
		}
		
		printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
		hit += curt_hit;
		tot += curt_tot;

		delete[] inf_label;
	}

	printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
}

void CRFModel::InitEvaluate(Config* conf, DataSet* test_data)
{
	this->conf = conf;
	this->test_data = test_data;
}

void CRFModel::Evalute()
{
	int ns = test_data->num_sample;
	int tot, hit;

    tot = hit = 0;

	FILE* fout = fopen(conf->pred_file.c_str(), "w");

	for (int s = 0; s < ns; s++)
	{
		DataSample* sample = test_data->sample[s];
		FactorGraph* factor_graph = new FactorGraph();
		
		int n = sample->num_node;
		int m = sample->num_edge;
		
		factor_graph->InitGraph(n, m, num_label);
		// Add edge info
		for (int i = 0; i < m; i++)
		{
			factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type]);
		}        
		factor_graph->GenPropagateOrder();

		factor_graph->ClearDataForMaxSum();

		for (int i = 0; i < n; i++)
		{
			double* p_lambda = lambda;

			for (int y = 0; y < num_label; y++)
			{
				double v = 1.0;
				for (int t = 0; t < sample->node[i]->num_attrib; t++)
					v *= exp(p_lambda[sample->node[i]->attrib[t]] * sample->node[i]->value[t]);
				factor_graph->SetVariableStateFactor(i, y, v);

				p_lambda += num_attrib_type;
			}
		}    

		factor_graph->MaxSumPropagation(conf->max_infer_iter);

		int* inf_label = new int[n];
		for (int i = 0; i < n; i++)
		{
			int ybest = -1;
			double vbest, v;

			for (int y = 0; y < num_label; y++)
			{
				v = factor_graph->var_node[i].state_factor[y];
				for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t++)
					v *= factor_graph->var_node[i].belief[t][y];
				if (ybest < 0 || v > vbest)
					ybest = y, vbest = v;
			}

			inf_label[i] = ybest;
		}

		int curt_tot, curt_hit;
		curt_tot = curt_hit = 0;
		for (int i = 0; i < n; i++)
		{   
			curt_tot ++;
			if (inf_label[i] == sample->node[i]->label) curt_hit++;
		}
		
		printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
		hit += curt_hit;
		tot += curt_tot;

		// to zz: just print inf_labe[0]
		for (int i = 0; i < n; i++)
		{
			fprintf(fout, "%s\n", train_data->label_dict.GetKeyWithId(inf_label[i]).c_str());
		}

		delete[] inf_label;
    }

	printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
	fclose(fout);
}

void CRFModel::Clean()
{
	if (lambda) delete[] lambda;
	if (sample_factor_graph) delete[] sample_factor_graph;

	for (int i = 0; i < num_edge_type; i++)
		delete func_list[i];
	delete[] func_list;
}

void CRFModel::LoadModel(const char* filename)
{
	FILE* fin = fopen(filename, "r");
	char buf[MAX_BUF_SIZE];
	vector<string> tokens;
	for (;;)
	{
		if (fgets(buf, MAX_BUF_SIZE, fin) == NULL)
			break;
		tokens = CommonUtil::StringTokenize(buf);
		if (tokens[0] == "#node")
		{
			int class_id = train_data->label_dict.GetIdConst(tokens[1]);
			int feat_id = train_data->attrib_dict.GetIdConst(tokens[2]);
			int pid = GetAttribParameterId(class_id, feat_id);
			double value = atof(tokens[3].c_str());
			lambda[pid] = value;
		}
		if (tokens[0] == "#edge")
		{
			int type_id = train_data->edge_type_dict.GetIdConst(tokens[1]);
			int id1 = train_data->label_dict.GetIdConst(tokens[2]);
			int id2 = train_data->label_dict.GetIdConst(tokens[3]);
			double value = atof(tokens[4].c_str());
			int i = GetEdgeParameterId(type_id, id1, id2);
			lambda[i] = value;
		}
	}
	fclose(fin);
	printf("Load %s finished.\n", filename);
}

void CRFModel::SaveModel(const char* filename)
{
	FILE* fout = fopen(filename, "w");
	for (int i = 0; i < num_label; i++)
	{
		string cl = train_data->label_dict.GetKeyWithId(i);
		for (int j = 0; j < num_attrib_type; j++)
		{
			string feature = train_data->attrib_dict.GetKeyWithId(j);
			int pid = GetAttribParameterId(i, j);
			fprintf(fout, "#node %s %s %f\n", cl.c_str(), feature.c_str(), lambda[pid]);
		}
	}
	for (int T = 0; T < num_edge_type; T++)
	{
		string c0 = train_data->edge_type_dict.GetKeyWithId(T);
		for (int i = 0; i < num_label; i++)
		{
			string c1 = train_data->label_dict.GetKeyWithId(i);
			for (int j = i; j < num_label; j++)
			{
				string c2 = train_data->label_dict.GetKeyWithId(j);
				int pid = GetEdgeParameterId(T, i, j);
				fprintf(fout, "#edge %s %s %s %f\n", c0.c_str(), c1.c_str(), c2.c_str(), lambda[pid]);
			}
		}
	}
	fclose(fout);
}

void CRFModel::Estimate(Config* conf)
{
	DataSet* dataset;

	dataset = new DataSet();
	dataset->LoadData(conf->train_file.c_str(), conf);
	dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

	printf("num_label = %d\n", dataset->num_label);
	printf("num_sample = %d\n", dataset->num_sample);
	printf("num_edge_type = %d\n", dataset->num_edge_type);
	printf("num_attrib_type = %d\n", dataset->num_attrib_type);
	
	InitTrain(conf, dataset);    

	printf("Start Training...\n");
	fflush(stdout);
	if (conf->method == "LBP") Train();
	else if (conf->method == "MH")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain(conf);
		SavePred("uncertainty.txt");
	}
	else if (conf->method == "MH1")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain1(conf);
		SavePred("uncertainty.txt");
	}
	else
	{
		printf("Method error!\n");
		return;
	}
	
	SaveModel(conf->dst_model_file.c_str());
}

void CRFModel::EstimateContinue(Config* conf)
{
	DataSet* dataset;

	dataset = new DataSet();
	dataset->LoadData(conf->train_file.c_str(), conf);
	dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

	printf("num_label = %d\n", dataset->num_label);
	printf("num_sample = %d\n", dataset->num_sample);
	printf("num_edge_type = %d\n", dataset->num_edge_type);
	printf("num_attrib_type = %d\n", dataset->num_attrib_type);
	
	InitTrain(conf, dataset);

	LoadModel(conf->src_model_file.c_str());

	if (conf->method == "LBP") Train();
	else if (conf->method == "MH")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain(conf);
		SavePred("uncertainty.txt");
	}
	else if (conf->method == "MH1")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain1(conf);
		SavePred("uncertainty.txt");
	}
	else
	{
		printf("Method error!\n");
		return;
	}

	SaveModel(conf->dst_model_file.c_str());
}

void CRFModel::Inference(Config* conf)
{
	DataSet* dataset;

	dataset = new DataSet();
	dataset->LoadData(conf->train_file.c_str(), conf);
	dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

	printf("num_label = %d\n", dataset->num_label);
	printf("num_sample = %d\n", dataset->num_sample);
	printf("num_edge_type = %d\n", dataset->num_edge_type);
	printf("num_attrib_type = %d\n", dataset->num_attrib_type);
	
	InitTrain(conf, dataset);
	LoadModel(conf->src_model_file.c_str());
	
	if (conf->method == "LBP")
	{
		InitEvaluate(conf, dataset);
		Evalute();
	}
	else if ((conf->method == "MH") || (conf->method == "MH1"))
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHEvaluate(conf->max_infer_iter, true);
	}
	else
	{
		printf("Method error!\n");
		return;
	}
}