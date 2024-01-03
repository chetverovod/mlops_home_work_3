import sys
import os
import os.path
import inspect
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
    cmd.append(["mv", "imagenette2-160/train/", "../datasets/"])
    cmd.append(["mv", "imagenette2-160/val/",   "../datasets/"])
    for c in cmd:
        print("Command: " + ' '.join(c))
        p = subprocess.check_output(c, universal_newlines=True)
        print('p = ' + p)


with mlflow.start_run():
    filename = "imagenette2-160.tgz"
    if os.path.isfile(filename):
            os.remove(filename)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/" + filename
    response = requests.get(url) 
    with open(filename, "wb") as file:
       file.write(response.content)
    unarchive()
    move_to_folders()
    mlflow.log_artifact(local_path="/home/igor/mlops_home_work_3/scripts/get_data.py",
                                    artifact_path="get_data code")

    mlflow.end_run()
