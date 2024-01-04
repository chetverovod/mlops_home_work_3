from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime as dt

args = {
        "owner": "admin",
        "start_date": dt.datetime(2024, 3, 1),
        "retries": 1,
        "retry_delays": dt.timedelta(minutes=1),
        "depends_on_past": False
        }

script_path = "/home/igor/mlops_home_work_3/scripts"

with DAG(dag_id='image_classification_random_train_size', default_args=args,
         schedule_interval='*/3 * * * *', tags=['image']) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command='python3 ' + script_path
                            + '/get_data.py',
                            dag=dag)

    prepare_data = BashOperator(task_id='prepare_data',
                                bash_command='python3 ' + script_path
                                + '/prepare.py',
                                dag=dag)
    train_test_split = BashOperator(task_id='train_test_split',
                                    bash_command='python3 ' + script_path
                                    + '/train.py',
                                    dag=dag)
    evaluate = BashOperator(task_id='evaluate',
                            bash_command='python3 ' + script_path
                            + '/evaluate.py',
                            dag=dag)
    # test_model = BashOperator(task_id='test_model',
    #       bash_command='python3 ' + script_path + '/test_model.py',
    #       dag=dag)

    get_data >> prepare_data >> train_test_split >> evaluate
