from joblib import dump
from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier
import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("model_training")


def load_images(data_frame, column_name):
    filelist = data_frame[column_name].to_list()
    image_list = imread_collection(filelist)
    return image_list


def load_labels(data_frame, column_name):
    label_list = data_frame[column_name].to_list()
    return label_list


def preprocess(image):
    resized = resize(image, (100, 100, 3))
    reshaped = resized.reshape((1, 30000))
    return reshaped


def load_data(data_path):
    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    raw_images = load_images(data_frame=df, column_name="filename")
    processed_images = [preprocess(image) for image in raw_images]
    data = np.concatenate(processed_images, axis=0)
    return data, labels


def main(repo_path):
    train_csv_path = repo_path / "prepared/train.csv"
    train_data, labels = load_data(train_csv_path)
    model_dir = repo_path / "model"
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    sgd = SGDClassifier(max_iter=100)
    trained_model = sgd.fit(train_data, labels)

    model_filename = model_dir / "model.joblib"
    dump(trained_model, model_filename)


with mlflow.start_run():
    repo_path = Path(__file__).parent.parent / "datasets"
    main(repo_path)

    local_path = "/home/igor/mlops_home_work_3/scripts/train.py"
    mlflow.log_artifact(local_path=local_path,
                        artifact_path="model_training code")
    mlflow.end_run()
