#include "Config.h"
#include "DataSet.h"
#include "CRFModel.h"

int main(int argc, char* argv[])
{
	// Load Configuartion
	Config* conf = new Config();
	if (! conf->LoadConfig(argc, argv))
	{
		conf->ShowUsage();
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

	return 0;
}