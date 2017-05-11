#include "CRFModel.h"
//#include "Transmitter.h"
#include "Constant.h"
#include <ctime>

/*--------------------------------------------------------------------------------*/
/*
extern "C" {
    // interface to LBFGS optimization written in FORTRAN
    extern void lbfgs_(int * n, int * m, double * x, double * f, double * g,
		       int * diagco, double * diag, int * iprint, double * eps,
		       double * xtol, double * w, int * iflag);		       
}
*/
/*--------------------------------------------------------------------------------*/

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
    for (int i = 0; i < num_feature; i ++)
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
    for (int y1 = 0; y1 < num_label; y1 ++)
        for (int y2 = y1; y2 < num_label; y2 ++)
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
    func_list = new EdgeFactorFunction*[ num_edge_type ];
    for (int i = 0; i < num_edge_type; i ++)
    {
        func_list[i] = new EdgeFactorFunction(num_label, p_lambda, &edge_feature_offset);
        p_lambda += num_edge_feature_each_type;
    }

    sample_factor_graph = new FactorGraph[num_sample];
    for (int s = 0; s < num_sample; s ++)
    {
        DataSample* sample = train_data->sample[s];

        int n = sample->num_node;
        int m = sample->num_edge;
        
        sample_factor_graph[s].InitGraph(n, m, num_label);

        // Add node info
        for (int i = 0; i < n; i ++)
        {
            sample_factor_graph[s].SetVariableLabel(i, sample->node[i]->label);
            sample_factor_graph[s].var_node[i].label_type = sample->node[i]->label_type;
        }   
        
        // Add edge info
        for (int i = 0; i < m; i ++)
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

    // Master node
    if (conf->my_rank == 0)
    {
        ///// Initilize all info

        // Data Varible         
        double  old_f = 0.0;

        // Variable for lbfgs optimization
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
        // Paramater estimation via L-BFGS
        num_iter = 0;

        double start_time, end_time;

        do {
            num_iter ++;

            if (conf->my_rank == 0)
            {
                start_time = clock() / (double)CLK_TCK;
            }
                        
            // Step A. Send lambda to all procs
            //Transmitter::Master_SendDoubleArray(lambda, num_feature, conf->num_procs);

            // Step B. Calc gradient and log-likehood of the local datas
            f = CalcGradient(gradient);

            // Step C. Collect gradient and log-likehood from all procs
            //Transmitter::Master_CollectGradientInfo(gradient, &f, num_feature, tmp_store, conf->num_procs);

            // Step 4. Opitmization by L-BFGS
            printf("[Iter %3d] log-likelihood : %.8lf\n", num_iter, f);
            fflush(stdout);

            // If diff of log-likelihood is small enough, break.
            if (fabs(old_f - f) < eps) break;
            old_f = f;
		
		    // Negate f and gradient vector because the LBFGS optimization below minimizes the ojective function while we would like to maximize it
            f *= -1;
            for (int i = 0; i < num_feature; i ++)
                gradient[i] *= -1;

            // Invoke L-BFGS

            if (conf->optimization_method == LBFGS)
            {
                //lbfgs_(&num_feature, &m_correlation, lambda, &f, gradient, &diagco, diag, iprint, &eps, &xtol, work_space, &iflag);

                // Checking after calling LBFGS
                if (iflag < 0) // LBFGS error
		        {
		            fprintf(stderr, "LBFGS routine encounters an error\n");
		            break;
		        }
            }
            else
            {
                // Normalize Graident
                double g_norm = 0.0;
                for (int i = 0; i < num_feature; i ++)
                    g_norm += gradient[i] * gradient[i];
                g_norm = sqrt(g_norm);
                
                if (g_norm > 1e-8)
                {
                    for (int i = 0; i < num_feature; i ++)
                        gradient[i] /= g_norm;
                }

                for (int i = 0; i < num_feature; i ++)
                    lambda[i] -= gradient[i] * conf->gradient_step;
                iflag = 1;
            }

			if ((num_iter % conf->eval_interval == 0) && (conf->my_rank == 0))
            {
                SelfEvaluate();
            }

            if (conf->my_rank == 0)
            {
                end_time = clock() / (double)CLK_TCK;

                FILE* ftime = fopen("time.out", "a");
                fprintf(ftime, "start_time = %.6lf\n", start_time);
                fprintf(ftime, "end_time = %.6lf\n", end_time);
                fprintf(ftime, "cost = %.6lf\n", end_time - start_time);
                
                fclose(ftime);

                printf("!!! Time cost = %.6lf\n", end_time - start_time);
                fflush(stdout);
            }
        } while (iflag != 0 && num_iter < conf->max_iter);



        //Transmitter::Master_SendQuit(conf->num_procs);

        delete[] tmp_store;

        delete[] work_space;
        delete[] diag;
    }
    /*else
    {
        bool done;

        while (1)
        {
            done = Transmitter::Slave_RecvDoubleArray(lambda, num_feature);            
            if (done) break;

            f = CalcGradient(gradient);

            Transmitter::Slave_SendGradientInfo(gradient, &f, num_feature);
        }
    }*/

    delete[] gradient;
}

double CRFModel::CalcGradient(double* gradient)
{
    double  f;
        
    // Initialize

    // If there is a square penalty, gradient should be initialized with (- lambda[i] / sigma^2). 
    // f should be accordingly modified as : -||lambda||^2/ (2*sigma^2)
    // note : should be added only in one procs (master)

    f = 0.0;
    for (int i = 0; i < num_feature; i ++)
    {
        //gradient[i] = - lambda[i] / conf->penalty_sigma_square;
        gradient[i] = 0; // no penalty
    }

    // Calculation
    for (int i = 0; i < num_sample; i ++)
    {
        // double t = CalcGradientForSample(train_data->sample[i], &sample_factor_graph[i], gradient);
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
    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y ++)
        {
            if (sample->node[i]->label_type == Enum::KNOWN_LABEL && y != sample->node[i]->label)
            {
                factor_graph->SetVariableStateFactor(i, y, 0);
            }
            else
            {
                double v = 1;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );                
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
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                gradient[ GetAttribParameterId(y, sample->node[i]->attrib[t]) ] += sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }

    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, a, b) ] += factor_graph->factor_node[i].marginal[a][b];
            }
    }

    //****************************************************************
    // Belief Propagation 2: labeled data are not given.
    //****************************************************************


    factor_graph->ClearDataForSumProduct();
    factor_graph->labeled_given = false;

    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y ++)
        {
            double v = 1;
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
            factor_graph->SetVariableStateFactor(i, y, v);
            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->BeliefPropagation(conf->max_infer_iter);
    factor_graph->CalculateMarginal();
        
    // calc gradient part : - E_{Y} f_i
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                gradient[ GetAttribParameterId(y, sample->node[i]->attrib[t]) ] -= sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, a, b) ] -= factor_graph->factor_node[i].marginal[a][b];
            }
    }
    
    // Calculate gradient & log-likelihood
    double f = 0.0, Z = 0.0;

    // \sum \lambda_i * f_i
    for (int i = 0; i < n; i ++)
    {
        int y = sample->node[i]->label;
        for (int t = 0; t < sample->node[i]->num_attrib; t ++)
            f += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t];
    }
    for (int i = 0; i < m; i ++)
    {
        int a = sample->node[ sample->edge[i]->a ]->label;
        int b = sample->node[ sample->edge[i]->b ]->label;        
        f += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)];
    }

    // calc log-likelihood
    //  using Bethe Approximation
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                Z += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                Z += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
            }
    }
    // Edge entropy
    for (int i = 0; i < m; i ++)
    {
        double h_e = 0.0;
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
                    h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
            }
        Z += h_e;
    }
    // Node entroy
    for (int i = 0; i < n; i ++)
    {
        double h_v = 0.0;
        for (int a = 0; a < num_label; a ++)
            if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
                h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
        Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
    }

    f -= Z;
    
//#ifdef DO_EVAL
    // Let's take a look of current accuracy

    factor_graph->ClearDataForMaxSum();
    factor_graph->labeled_given = true;

        for (int i = 0; i < n; i ++)
        {
            double* p_lambda = lambda;

            for (int y = 0; y < num_label; y ++)
            {
                double v = 1.0;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
                factor_graph->SetVariableStateFactor(i, y, v);

                p_lambda += num_attrib_type;
            }
        }    

        factor_graph->MaxSumPropagation(conf->max_infer_iter);

        int* inf_label = new int[n];
		double** label_prob = new double*[num_label];
		for (int p = 0; p < num_label; p ++)
			label_prob[p] = new double[n];

        for (int i = 0; i < n; i ++)
        {
            int ybest = -1;
            double vbest, v;
			double vsum = 0.0;
            for (int y = 0; y < num_label; y ++)
            {
                v = factor_graph->var_node[i].state_factor[y];
                for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t ++)
                    v *= factor_graph->var_node[i].belief[t][y];
                if (ybest < 0 || v > vbest)
                    ybest = y, vbest = v;

				label_prob[y][i] = v;
				vsum += v;
            }

            inf_label[i] = ybest;

			for (int y = 0; y < num_label; y ++)
				label_prob[y][i] /= vsum;
        }

    int hit = 0, miss = 0;
    int hitu = 0, missu = 0;

	int cnt[10][10];
	int ucnt[10][10];

	memset(cnt, 0, sizeof(cnt));
	memset(ucnt, 0, sizeof(ucnt));
	FILE *pred_out = fopen("pred.txt", "w");

    for (int i = 0; i < n; i ++)
    {
        /*
        int ymax = 0;
        for (int y = 0; y < num_label; y ++)
            if (factor_graph->var_node[i].marginal[y] > factor_graph->var_node[i].marginal[ymax])
                ymax = y;
                */

		fprintf(pred_out, "%d\n", inf_label[i]);
        if (inf_label[i] == sample->node[i]->label)
            hit ++;
        else
            miss ++;

		cnt[ inf_label[i] ][ sample->node[i]->label ] ++;

        if (sample->node[i]->label_type == Enum::UNKNOWN_LABEL)
        {
            if (inf_label[i] == sample->node[i]->label)
                hitu ++;
            else
                missu ++;

			ucnt[ inf_label[i] ][ sample->node[i]->label ] ++;
        }
    }
	fclose(pred_out);
    //printf("HIT = %4d, MISS = %4d, All_Accuracy = %.5lf Unknown_Accuracy = %.5lf\n", hit, miss, (double)hit / (hit + miss), (double)hitu / (hitu + missu));

    if (conf->my_rank != 0)
    {        
        // Slave Send Result
        /*int dat[12];
        dat[0] = hit; dat[1] = hitu;
        dat[2] = miss; dat[3] = missu;
        dat[4] = cnt[0][0]; dat[5] = cnt[0][1]; dat[6] = cnt[1][0]; dat[7] = cnt[1][1];
        dat[8] = ucnt[0][0]; dat[9] = ucnt[0][1]; dat[10] = ucnt[1][0]; dat[11] = ucnt[1][1];

        Transmitter::Slave_SendIntArr(dat, 12);*/
    }
    else
    {
        int dat[12];
        memset(dat, 0, sizeof(dat));
        //Transmitter::Master_CollectIntArr(dat, 12, conf->num_procs);

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

    }

    
	FILE* fprob = fopen("uncertainty.txt", "w");
	for (int i = 0; i < n; i ++)
	{
		if (sample->node[i]->label_type == Enum::KNOWN_LABEL)
		{
			for (int y = 0; y < num_label; y ++)
				fprintf(fprob, "-1 ");
			fprintf(fprob, "\n");
		}
		else
		{
			for (int y = 0; y < num_label; y ++)
				fprintf(fprob, "%.4lf ", label_prob[y][i]);
			fprintf(fprob, "\n");
		}
	}
	fclose(fprob);
    

	delete[] inf_label;
	for (int y = 0; y < num_label; y ++)
		delete[] label_prob[y];
	delete[] label_prob;
//#endif

    return f;
}

double CRFModel::CalcGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient)
{
    factor_graph->ClearDataForSumProduct();

    // Set state_factor
    int n = sample->num_node;
    int m = sample->num_edge;

    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y ++)
        {
            double v = 1;
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
            factor_graph->SetVariableStateFactor(i, y, v);
            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->BeliefPropagation(conf->max_infer_iter);
    factor_graph->CalculateMarginal();
    
    // Calculate gradient & log-likelihood
    double f = 0.0, Z = 0.0;

    // \sum \lambda_i * f_i
    for (int i = 0; i < n; i ++)
    {
        int y = sample->node[i]->label;
        for (int t = 0; t < sample->node[i]->num_attrib; t ++)
            f += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t];
    }
    for (int i = 0; i < m; i ++)
    {
        int a = sample->node[ sample->edge[i]->a ]->label;
        int b = sample->node[ sample->edge[i]->b ]->label;        
        f += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)];
    }

    // calc log-likelihood
    //  using Bethe Approximation
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                Z += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                Z += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
            }
    }
    // Edge entropy
    for (int i = 0; i < m; i ++)
    {
        double h_e = 0.0;
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
                    h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
            }
        Z += h_e;
    }
    // Node entroy
    for (int i = 0; i < n; i ++)
    {
        double h_v = 0.0;
        for (int a = 0; a < num_label; a ++)
            if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
                h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
        Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
    }

    f -= Z;
    fflush(stdout);

    // calc gradient
    for (int i = 0; i < n; i ++)
        for (int t = 0; t < sample->node[i]->num_attrib; t ++)
            gradient[ GetAttribParameterId(sample->node[i]->label, sample->node[i]->attrib[t]) ] += sample->node[i]->value[t];
    for (int i = 0; i < m; i ++)
        gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, 
                                     sample->node[sample->edge[i]->a]->label, 
                                     sample->node[sample->edge[i]->b]->label) ] += 1.0;

    // - expectation
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                gradient[ GetAttribParameterId(y, sample->node[i]->attrib[t]) ] -= sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, a, b) ] -= factor_graph->factor_node[i].marginal[a][b];
            }
    }

    return f;
}

void CRFModel::SelfEvaluate()
{
    int ns = train_data->num_sample;
    int tot, hit;

    tot = hit = 0;
    for (int s = 0; s < ns; s ++)
    {
        DataSample* sample = train_data->sample[s];
        FactorGraph* factor_graph = &sample_factor_graph[s];
        
        int n = sample->num_node;
        int m = sample->num_edge;
        
        factor_graph->InitGraph(n, m, num_label);
        // Add edge info
        for (int i = 0; i < m; i ++)
        {
            factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type]);
        }        
        factor_graph->GenPropagateOrder();

        factor_graph->ClearDataForMaxSum();

        for (int i = 0; i < n; i ++)
        {
            double* p_lambda = lambda;

            for (int y = 0; y < num_label; y ++)
            {
                double v = 1.0;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
                factor_graph->SetVariableStateFactor(i, y, v);

                p_lambda += num_attrib_type;
            }
        }    

        factor_graph->MaxSumPropagation(conf->max_infer_iter);

        int* inf_label = new int[n];
        for (int i = 0; i < n; i ++)
        {
            int ybest = -1;
            double vbest, v;

            for (int y = 0; y < num_label; y ++)
            {
                v = factor_graph->var_node[i].state_factor[y];
                for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t ++)
                    v *= factor_graph->var_node[i].belief[t][y];
                if (ybest < 0 || v > vbest)
                    ybest = y, vbest = v;
            }

            inf_label[i] = ybest;
        }

        int curt_tot, curt_hit;
        curt_tot = curt_hit = 0;
        for (int i = 0; i < n; i ++)
        {   
            curt_tot ++;
            if (inf_label[i] == sample->node[i]->label) curt_hit ++;
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

    for (int s = 0; s < ns; s ++)
    {
        DataSample* sample = test_data->sample[s];
        FactorGraph* factor_graph = new FactorGraph();
        
        int n = sample->num_node;
        int m = sample->num_edge;
        
        factor_graph->InitGraph(n, m, num_label);
        // Add edge info
        for (int i = 0; i < m; i ++)
        {
            factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, func_list[sample->edge[i]->edge_type]);
        }        
        factor_graph->GenPropagateOrder();

        factor_graph->ClearDataForMaxSum();

        for (int i = 0; i < n; i ++)
        {
            double* p_lambda = lambda;

            for (int y = 0; y < num_label; y ++)
            {
                double v = 1.0;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
                factor_graph->SetVariableStateFactor(i, y, v);

                p_lambda += num_attrib_type;
            }
        }    

        factor_graph->MaxSumPropagation(conf->max_infer_iter);

        int* inf_label = new int[n];
        for (int i = 0; i < n; i ++)
        {
            int ybest = -1;
            double vbest, v;

            for (int y = 0; y < num_label; y ++)
            {
                v = factor_graph->var_node[i].state_factor[y];
                for (int t = 0; t < factor_graph->var_node[i].neighbor.size(); t ++)
                    v *= factor_graph->var_node[i].belief[t][y];
                if (ybest < 0 || v > vbest)
                    ybest = y, vbest = v;
            }

            inf_label[i] = ybest;
        }

        int curt_tot, curt_hit;
        curt_tot = curt_hit = 0;
        for (int i = 0; i < n; i ++)
        {   
            curt_tot ++;
            if (inf_label[i] == sample->node[i]->label) curt_hit ++;
        }
        
        printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
        hit += curt_hit;
        tot += curt_tot;

        // to zz: just print inf_labe[0]
        for (int i = 0; i < n; i ++)
        {
            fprintf(fout, "%d\n", inf_label[i]);
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

    for (int i = 0; i < num_edge_type; i ++)
        delete func_list[i];
    delete[] func_list;
}

void CRFModel::SaveModel(const char* file_name)
{
    FILE* fout = fopen(file_name, "w");
    fprintf(fout, "%d\n", num_feature);
    for (int i = 0; i < num_feature; i ++)
        fprintf(fout, "%.6lf\n", lambda[i]);
    fclose(fout);
}

void CRFModel::LoadModel(const char* file_name)
{
    FILE* fin = fopen(file_name, "r");
    int num_feature_saved;
    fscanf(fin, "%d", &num_feature_saved);
    if (num_feature_saved != num_feature)
    {
        fprintf(stderr, "[error] model from %s not match\n", file_name);
        return;
    }
    for (int i = 0; i < num_feature; i ++)
        fscanf(fin, "%lf", &lambda[i]);
    fclose(fin);
}

void CRFModel::MakeEvaluate(Config* conf, DataSet* dataset)
{
    if (conf->my_rank == 0)
    {
        //GlobalDataSet* g_testdata = new GlobalDataSet();
        DataSet* testdata = new DataSet();        

        testdata->LoadDataWithDict(conf->test_file.c_str(), conf, dataset->label_dict, dataset->attrib_dict, dataset->edge_type_dict);
        /*testdata->sample = g_testdata->sample;
        testdata->num_sample = g_testdata->sample.size();
        testdata->num_edge_type = g_testdata->num_edge_type;
        testdata->num_attrib_type = g_testdata->num_attrib_type;
        testdata->num_label = g_testdata->num_label;

        for (int i = 0; i < testdata->num_sample; i ++)
        {        
            testdata->sample[i]->num_node = testdata->sample[i]->node.size();
            testdata->sample[i]->num_edge = testdata->sample[i]->edge.size();        

            for (int t = 0; t < testdata->sample[i]->num_node; t ++)
                testdata->sample[i]->node[t]->num_attrib = testdata->sample[i]->node[t]->attrib.size();
        }*/

        //DEBUG__AddLinearEdge(testdata);

		InitEvaluate(conf, testdata);
        Evalute();
    }
}

void CRFModel::Estimate(Config* conf)
{
    DataSet* dataset;

    if (conf->my_rank == 0) // master
    {
        dataset = new DataSet();
        dataset->LoadData(conf->train_file.c_str(), conf);
        dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

        // Assign jobs
        //dataset = Transmitter::AssignJobs(g_dataset, conf->num_procs);
    }
    else
    {
        // Get jobs
        //dataset = Transmitter::GetJobs();
    }

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
		SavePred(conf->pred_file.c_str());
	}
	else if (conf->method == "MH1")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain1(conf);
		SavePred(conf->pred_file.c_str());
	}
	else
	{
		printf("Method error!\n");
		return;
	}
    
    if (conf->my_rank == 0)
        SaveModel(conf->dst_model_file.c_str());
      
    //MakeEvaluate(conf, g_dataset, model);
}

void CRFModel::EstimateContinue(Config* conf)
{
    //GlobalDataSet* g_dataset;
    DataSet* dataset;

    if (conf->my_rank == 0) // master
    {
        dataset = new DataSet();
        dataset->LoadData(conf->train_file.c_str(), conf);
        dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

        // Assign jobs
        //dataset = Transmitter::AssignJobs(g_dataset, conf->num_procs);
    }
    else
    {
        // Get jobs
        //dataset = Transmitter::GetJobs();
    }

    //DEBUG__AddLinearEdge(dataset);

    printf("num_label = %d\n", dataset->num_label);
    printf("num_sample = %d\n", dataset->num_sample);
    printf("num_edge_type = %d\n", dataset->num_edge_type);
    printf("num_attrib_type = %d\n", dataset->num_attrib_type);
    
    InitTrain(conf, dataset);

    if (conf->my_rank == 0)
        LoadModel(conf->src_model_file.c_str());

    if (conf->method == "LBP") Train();
	else if (conf->method == "MH")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain(conf);
		SavePred(conf->pred_file.c_str());
	}
	else if (conf->method == "MH1")
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHTrain1(conf);
		SavePred(conf->pred_file.c_str());
	}
	else
	{
		printf("Method error!\n");
		return;
	}

    if (conf->my_rank == 0)

        SaveModel(conf->dst_model_file.c_str());
    
    //MakeEvaluate(conf, g_dataset, model);
}

void CRFModel::Inference(Config* conf)
{
    //GlobalDataSet* g_dataset;
    DataSet* dataset;

    if (conf->my_rank == 0) // master
    {
        dataset = new DataSet();
        dataset->LoadData(conf->train_file.c_str(), conf);
        dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

        // Assign jobs
        //dataset = Transmitter::AssignJobs(g_dataset, conf->num_procs);
    }
    else
    {
        // Get jobs
        //dataset = Transmitter::GetJobs();
    }

    //DEBUG__AddLinearEdge(dataset);

    printf("num_label = %d\n", dataset->num_label);
    printf("num_sample = %d\n", dataset->num_sample);
    printf("num_edge_type = %d\n", dataset->num_edge_type);
    printf("num_attrib_type = %d\n", dataset->num_attrib_type);
    
    InitTrain(conf, dataset);
    if (conf->my_rank == 0)
        LoadModel(conf->src_model_file.c_str());
    
    if (conf->method == "LBP") MakeEvaluate(conf, dataset);
	else if ((conf->method == "MH") || (conf->method == "MH1"))
	{
		if (conf->state_file != "") 
			LoadStateFile(conf->state_file.c_str());
		MHEvaluate(conf->max_infer_iter);
		SavePred(conf->pred_file.c_str());
	}
	else
	{
		printf("Method error!\n");
		return;
	}
}