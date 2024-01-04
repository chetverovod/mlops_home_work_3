from pathlib import Path
import os
import shutil
import pandas as pd
import mlflow
import random

FOLDERS_TO_LABELS = {"n03445777": "golf ball", "n03888257": "parachute"}

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("prepare_data_for_model_training")


def get_files_and_labels(source_path, ratio:float=1.):
    images = []
    labels = []
    for image_path in source_path.rglob("*/*.JPEG"):
        filename = image_path.absolute()
        folder = image_path.parent.name
        if folder in FOLDERS_TO_LABELS:
            images.append(filename)
            label = FOLDERS_TO_LABELS[folder]
            labels.append(label)
    sz = int(len(images) * ratio)        
    print("partition size:", sz)
    im, la = zip(*random.sample(list(zip(images, labels)), sz))
    return  im, la


def save_as_csv(filenames, labels, destination):
    data_dictionary = {"filename": filenames, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)
    data_frame.to_csv(destination)


def main(repo_path: Path):
    data_path = repo_path
    train_path = data_path / "train"
    test_path = data_path / "val"

    partition = random.uniform(0, 1)

    train_files, train_labels = get_files_and_labels(train_path, partition)
    test_files, test_labels = get_files_and_labels(test_path, 1.)
    print("train_partition", partition)
    mlflow.log_param("train_partition", partition)
    mlflow.log_param("train_size", len(test_labels))

    prepared = data_path / "prepared"
    if os.path.isdir(prepared):
        shutil.rmtree(prepared)
    os.mkdir(prepared)
    save_as_csv(train_files, train_labels, prepared / "train.csv")
    save_as_csv(test_files, test_labels, prepared / "test.csv")


with mlflow.start_run():
    repo_path = Path(os.path.dirname(os.path.dirname(Path(__file__).resolve()))) / "datasets"
    print("repo_path =" , repo_path)
    main(repo_path)

    local_path = "/home/igor/mlops_home_work_3/scripts/prepare.py"
    mlflow.log_artifact(local_path=local_path, artifact_path="prepare code")
    mlflow.end_run()
