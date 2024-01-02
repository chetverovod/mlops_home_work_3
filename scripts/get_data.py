import sys
import subprocess
import requests
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_data form Amazon")

def unarchive():
        function_name = inspect.stack()[0][3]
        print("App function: " + function_name)
        cmd = ["tar", "zxvf", "./imagenette2-160.tgz"]
        print("Command: " + ' '.join(cmd))
        p = subprocess.check_output(cmd, universal_newlines=True)
        print('p = ' + p)


def move_to_folders():
    cmd = []
    cmd[0] = ["mv", "imagenette2-160/train/", "../datasets/"]
    cmd[1] = ["mv", "imagenette2-160/val/",   "../datasets/"]
    for c in cmd:
        print("Command: " + ' '.join(c))
        p = subprocess.check_output(c, universal_newlines=True)
        print('p = ' + p)


with mlflow.start_run():
    data = requests.get("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz")
    unarchive()
    move_to_folders()
    mlflow.log_artifact(local_path="/home/pyretttt/repos/mlops3/scripts/get_data.py",
                                    artifact_path="get_data code")

    mlflow.end_run()
