from joblib import load
import json
from pathlib import Path
import os
import shutil
from sklearn.metrics import accuracy_score

from train import load_data

import mlflow

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("model_training")

accuracy = ""


def main(repo_path):
    test_csv_path = repo_path / "prepared/test.csv"
    test_data, labels = load_data(test_csv_path)
    model = load(repo_path / "model/model.joblib")
    predictions = model.predict(test_data)
    accuracy = accuracy_score(labels, predictions)
    metrics = {"accuracy": accuracy}

    metrics_dir = repo_path / "metrics"
    if os.path.isdir(metrics_dir):
        shutil.rmtree(metrics_dir)
    os.mkdir(metrics_dir)
    accuracy_path = metrics_dir / "accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))
    return accuracy


with mlflow.start_run():
    #repo_path = Path(__file__).parent.parent / "datasets"
    repo_path = Path(os.path.dirname(os.path.dirname(Path(__file__).resolve()))) / "datasets"
    a = main(repo_path)

    local_path = "/home/igor/mlops_home_work_3/scripts/evaluate.py"
    mlflow.log_artifact(local_path=local_path,
                        artifact_path="model_evaluation code")
    print("accuracy", a)
    mlflow.log_metric("accuracy", a)
    mlflow.end_run()
