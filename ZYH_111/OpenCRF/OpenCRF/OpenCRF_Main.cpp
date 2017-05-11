//#include "mpi.h"

#include "Config.h"
#include "DataSet.h"
//#include "Transmitter.h"
#include "CRFModel.h"

int main(int argc, char* argv[])
{
    int         my_rank = 0;
    int         num_procs = 1;

    // Initialize MPI environment
    //MPI_Init(&argc, &argv);
    //MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Load Configuartion
    Config* conf = new Config();
    if (! conf->LoadConfig(my_rank, num_procs, argc, argv))
    {
        conf->ShowUsage();
        //MPI_Finalize();
        exit( 0 );
    }

	CRFModel *model = new CRFModel();

    if (conf->task == "-est")
    {
        model->Estimate(conf);
    }
    else if (conf->task == "-estc")
    {
        model->EstimateContinue(conf);
    }
    else if (conf->task == "-inf")
    {
        model->Inference(conf);
    }
    else
    {
        Config::ShowUsage();
    }

    //MPI_Finalize();
}