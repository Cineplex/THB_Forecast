import argparse
import sys
import os

# Add the project root to the python path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.de.raw_data import load_to_raw_data
from src.de.cleaning_data import clean_and_load_cleaning_data
from src.de.feature_data import build_feature_data
from src.ds.models.train_xgboost import train as train_xgboost
from src.ds.models.train_lstm import train as train_lstm
from src.ds.models.train_tft import train as train_tft

def run_de_pipeline():
    # 1. Data Engineering
    print("\n" + "="*50)
    print("📦 STAGE 1: Data Engineering")
    print("="*50)
    
    try:
        print("\n[1/3] Ingesting Raw Data...")
        load_to_raw_data()
        
        print("\n[2/3] Cleaning Data...")
        clean_and_load_cleaning_data()
        
        print("\n[3/3] Building Features...")
        build_feature_data()
        
    except Exception as e:
        print(f"\n❌ Data Engineering Pipeline Failed: {e}")
        raise e

def run_ds_pipeline():
    # 2. Data Science (Modeling)
    print("\n" + "="*50)
    print("🤖 STAGE 2: Model Training")
    print("="*50)
    
    try:
        print("\n[1/3] Training XGBoost...")
        train_xgboost()
        
        print("\n[2/3] Training LSTM...")
        train_lstm()
        
        print("\n[3/3] Training TFT...")
        train_tft()
        
    except Exception as e:
        print(f"\n❌ Model Training Pipeline Failed: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="THB Forecast Project Runner")
    parser.add_argument('-all', action='store_true', help="Run the entire pipeline (Data + Models)")
    parser.add_argument('-de', action='store_true', help="Run only Data Engineering pipeline")
    parser.add_argument('-ds', action='store_true', help="Run only Data Science (Modeling) pipeline")
    
    args = parser.parse_args()
    
    # Logic to determine what to run
    run_de = args.all or args.de
    run_ds = args.all or args.ds
    
    if not run_de and not run_ds:
        parser.print_help()
        return

    print("🚀 Starting Project Pipeline...")
    
    try:
        if run_de:
            run_de_pipeline()
            
        if run_ds:
            run_ds_pipeline()
            
        print("\n" + "="*50)
        print("✅ Pipeline Execution Completed Successfully!")
        print("="*50)
        
    except Exception:
        # Error is already printed in the sub-functions
        print("\n❌ Pipeline Execution Stopped due to Error.")

if __name__ == "__main__":
    main()
