from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ensure src is in python path (mapped in docker-compose)
sys.path.append('/opt/airflow/dags/repo')

# Import your existing functions
from src.de.raw_data import load_to_raw_data
from src.de.cleaning_data import clean_and_load_cleaning_data
from src.de.feature_data import build_feature_data
from src.de.date import load_calendar_date

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'project_forecast',
    default_args=default_args,
    description='Automated Data Pipeline for Project Forecast',
    schedule_interval='*/10 * * * *',  # Run every 10 minutes
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['project_forecast', 'data_engineering'],
) as dag:

    t0_calendar = PythonOperator(
        task_id='generate_calendar',
        python_callable=load_calendar_date,
    )

    t1_extract = PythonOperator(
        task_id='extract_raw_data',
        python_callable=load_to_raw_data,
    )

    t2_process = PythonOperator(
        task_id='clean_data',
        python_callable=clean_and_load_cleaning_data,
    )

    t3_feature = PythonOperator(
        task_id='build_features',
        python_callable=build_feature_data,
    )

    # Define dependencies
    # 1. Extract -> Clean
    t1_extract >> t2_process
    
    # 2. Clean + Calendar -> Feature
    [t2_process, t0_calendar] >> t3_feature
