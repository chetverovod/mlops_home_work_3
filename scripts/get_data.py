import os
import os.path
import shutil
import inspect
import subprocess
import requests
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_data_form_Amazon")


def unarchive(arch_name):
    function_name = inspect.stack()[0][3]
    print("App function: " + function_name)
    cmd = ["tar", "zxvf", "./" + arch_name]
    print("Command: " + ' '.join(cmd))
    p = subprocess.check_output(cmd, universal_newlines=True)
    print('p = ' + p)


def move_to_folders(path):
    cmd = []
    cmd.append(["mv", path + "/train/", "../datasets/"])
    cmd.append(["mv", path + "/val/",   "../datasets/"])
    dest_train = "../datasets/train/"
    dest_val = "../datasets/val/"

    for name in [dest_train, dest_val]:
        if os.path.isdir(name):
            shutil.rmtree(name)
    for c in cmd:
        print("Command: " + ' '.join(c))
        p = subprocess.check_output(c, universal_newlines=True)
        print('p = ' + p)
    if os.path.isdir(path):
        shutil.rmtree(path)


def new_func(url):
    return requests.get(url)

with mlflow.start_run():
    set_name = "imagenette2-160"
    filename = set_name + ".tgz"
    #dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    if os.path.isfile(filename):
        os.remove(filename)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/" + filename
    print("url =", url)
    response = new_func(url) 
    with open(filename, "wb") as file:
        print("file writing:", filename)
        file.write(response.content)

    unarchive(filename)
    move_to_folders(set_name)
     
    #local_path = '/home/igor/Plastov/MLOPs_sem3/home_work3/mlops_home_work_3/scripts'
    local_path = "/home/igor/mlops_home_work_3/scripts/get_data.py"
    mlflow.log_artifact(local_path=local_path, artifact_path="get_data code")
    mlflow.end_run()
