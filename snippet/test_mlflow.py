
import subprocess
import mlflow
import numpy as np

if __name__ == "__main__" : 

    client = mlflow.tracking.MlflowClient()
    # create exp
    experiment = "experiment"
    try:
        exp_id = client.create_experiment(experiment)
    except:
        exp_id = client.get_experiment_by_name(experiment).experiment_id
    
    # MLFlow system tags 
    # - https://mlflow.org/docs/latest/tracking.html?highlight=commit#system-tags
    tags = {"mlflow.source.git.commit" : subprocess.check_output("git rev-parse HEAD".split()).strip().decode("utf-8") }
    run = client.create_run(exp_id, tags=tags)
    
    client.log_param(run.info.run_id, "a", "aaaaa")

    for i in range(10):
        client.log_metric(run.info.run_id, "loss", i * np.random.randn(1), step=i)