import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from src.de.utils.db_connector import get_engine

warnings.filterwarnings('ignore')

# --- ⚙️ Configuration (ล็อคค่าให้ผลนิ่งที่สุด) ---
SEED = 42  # เลขมงคล (ล็อคความสุ่ม)
VAL_START_DATE = "2023-01-01"
TEST_START_DATE = "2024-01-01"

BEST_PARAMS = {
    'n_estimators': 3000, 
    'learning_rate': 0.005, 
    'max_depth': 9, 
    'min_child_weight': 1,
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 
    'reg_lambda': 1.0, 
    'n_jobs': -1, 
    'early_stopping_rounds': 50,
    'device': "cuda",         # ใช้ GPU (ถ้าไม่มี แก้เป็น "cpu")
    'enable_categorical': True,
    'random_state': SEED,     # <--- ล็อคผล 1
    'seed': SEED              # <--- ล็อคผล 2
}

# --- 1. Load Data ---
def load_feature_data():
    engine = get_engine()
    if not engine: return None
    
    print("📥 Loading data from Feature Layer...")
    # ดึงทั้งหมด (ไม่กรอง version)
    query = "SELECT * FROM feature_data ORDER BY record_date ASC"


    df = pd.read_sql(query, engine)
    df['record_date'] = pd.to_datetime(df['record_date'])
    df = df.set_index('record_date')
    return df

# --- 2. Metrics Calculation ---
def calculate_metrics(model, X, y_true_price, y_lag1):
    # Predict Diff
    pred_diff = model.predict(X)
    
    # Reconstruct Price: Price(t) = Price(t-1) + Diff
    pred_price = y_lag1 + pred_diff
    
    # Metrics
    mape = mean_absolute_percentage_error(y_true_price, pred_price)
    mae = mean_absolute_error(y_true_price, pred_price)
    
    # Direction Accuracy
    # เทียบทิศทางจริง (y_true - y_lag1) กับ ทิศทางที่ทำนาย (pred_diff)
    actual_diff = y_true_price - y_lag1
    
    # กรองวันที่ราคาไม่เปลี่ยน (0) ออก เพื่อความแฟร์
    nonzero_idx = (actual_diff != 0) & (pred_diff != 0)
    if np.sum(nonzero_idx) > 0:
        dir_acc = (np.sign(actual_diff[nonzero_idx]) == np.sign(pred_diff[nonzero_idx])).mean()
    else:
        dir_acc = 0.0
        
    return mape, mae, dir_acc, pred_price

def train():
    # --- Load ---
    df = load_feature_data()
    if df is None or df.empty: return

    # --- Select Features ---
    feature_cols = [
        'gold', 'oil', 'bond_yield', 'dxy', 'sp500', 'set_index', 
        'rsi', 'macd', 'pct_change',
        'volatility_5', 'volatility_20',
        'gold_oil_ratio', 'bond_dxy_ratio', 'dist_sma20',
        'lag_1', 'lag_7',
        'day_of_week', 'month', 'is_holiday_th'
    ]
    features = [c for c in feature_cols if c in df.columns] # Safety check
    target = 'target_diff'

    # Convert Categories for XGBoost
    cat_cols = ['day_of_week', 'month', 'is_holiday_th']
    for c in cat_cols:
        if c in df.columns: df[c] = df[c].astype('category')

    X = df[features]
    y = df[target]
    
    # --- Split Data ---
    mask_train = X.index < VAL_START_DATE
    mask_val = (X.index >= VAL_START_DATE) & (X.index < TEST_START_DATE)
    mask_test = X.index >= TEST_START_DATE
    
    X_train, y_train = X.loc[mask_train], y.loc[mask_train]
    X_val, y_val = X.loc[mask_val], y.loc[mask_val]
    X_test, y_test = X.loc[mask_test], y.loc[mask_test]
    
    # --- Train ---
    print(f"🚀 Training XGBoost (Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)})...")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   👉 GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️ GPU not detected by PyTorch (XGBoost might still work if configured).")
    except ImportError:
        pass

    try:
        model = XGBRegressor(**BEST_PARAMS)
        model.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            verbose=False
        )
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return
    
    # --- Evaluate (Deep Check) ---
    # เราต้องดึงราคาจริง (thb_usd) จาก DB มาเทียบ (ไม่ใช่ Diff)
    # แต่ใน Feature Table เราไม่ได้ดึง thb_usd มา (เรามีแต่ target_diff และ lag_1)
    # ดังนั้น: Actual Price(t) = Lag1(t) + Target_Diff(t)
    
    # 1. Train Metrics
    y_true_train = X_train['lag_1'] + y_train
    train_mape, _, train_acc, train_pred_price = calculate_metrics(model, X_train, y_true_train, X_train['lag_1'])
    
    # 2. Val Metrics
    y_true_val = X_val['lag_1'] + y_val
    val_mape, _, val_acc, val_pred_price = calculate_metrics(model, X_val, y_true_val, X_val['lag_1'])
    
    # 3. Test Metrics (Final Exam)
    y_true_test = X_test['lag_1'] + y_test
    test_mape, test_mae, test_acc, test_pred_price = calculate_metrics(model, X_test, y_true_test, X_test['lag_1'])

    # --- Report ---
    print("\n" + "="*40)
    print("📊 DETAILED PERFORMANCE REPORT")
    print("="*40)
    print(f"{'Set':<10} | {'MAPE (%)':<10} | {'Accuracy (%)':<10}")
    print("-" * 36)
    print(f"{'Train':<10} | {train_mape*100:.4f}     | {train_acc*100:.2f}")
    print(f"{'Val':<10}   | {val_mape*100:.4f}     | {val_acc*100:.2f}")
    print(f"{'Test':<10}  | {test_mape*100:.4f}     | {test_acc*100:.2f}")
    print("-" * 36)
    
    # Overfitting Analysis
    gap = test_mape - train_mape
    print(f"🔍 Analysis:")
    print(f"   Gap (Test-Train): {gap*100:.4f}%")
    if gap > 0.2:
        print("   Status: ⚠️ Possible Overfitting (Test error much higher)")
    else:
        print("   Status: ✅ Good Fit (Model generalizes well)")
        
    print(f"   Final MAE: {test_mae:.4f} THB (Avg Error)")
    print("="*40)

    # --- Save Model ---
    try:
        import os
        import joblib
        from datetime import datetime
        
        # สร้างโฟลเดอร์ save_models ถ้ายังไม่มี
        model_dir = os.path.join("src", "ds", "models", "save_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # รวมทุกอย่างในไฟล์เดียว
        model_package = {
            'model': model,
            'features': feature_cols,
            'dtypes': X_train.dtypes.to_dict(), # <--- Save dtypes for inference consistency
            'metadata': {
                'model_type': 'XGBoost',
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_mape': float(test_mape),
                'test_mae': float(test_mae),
                'test_accuracy': float(test_acc),
                'val_start_date': VAL_START_DATE,
                'test_start_date': TEST_START_DATE,
                'seed': SEED,
                'n_estimators': BEST_PARAMS['n_estimators'],
                'learning_rate': BEST_PARAMS['learning_rate']
            }
        }
        
        # บันทึกเป็นไฟล์เดียว
        model_path = os.path.join(model_dir, "xgboost_model.pkl")
        joblib.dump(model_package, model_path)
        
        print(f"\n💾 Model Saved Successfully!")
        print(f"   Path: {model_path}")
        print(f"   Includes: model + features + metadata")
        
    except Exception as e:
        print(f"\n❌ Model Save Error: {e}")

    # --- Plotting ---
    try:
        # Prepare DataFrames for plotting
        train_final = pd.DataFrame({'record_date': X_train.index, 'pred_price': train_pred_price})
        val_final = pd.DataFrame({'record_date': X_val.index, 'pred_price': val_pred_price})
        test_final = pd.DataFrame({'record_date': X_test.index, 'pred_price': test_pred_price})
        
        # Reset index to make record_date a column if it's currently the index (it is)
        df_plot = df.reset_index()

        # Plotting Logic
        plt.figure(figsize=(18, 8))
        
        # 1. Actual Price (เส้นจริง)
        plt.plot(df_plot['record_date'], df_plot['thb_usd'], label='Actual Price', color='black', alpha=0.3, linewidth=1)
        
        # 2. Train Prediction (สีเขียว)
        plt.plot(train_final['record_date'], train_final['pred_price'], label='Train (Learn)', color='green', alpha=0.8, linewidth=1)
        
        # 3. Validation Prediction (สีส้ม)
        plt.plot(val_final['record_date'], val_final['pred_price'], label='Validation (Tune)', color='orange', alpha=0.9, linewidth=1.5)
        
        # 4. Test Prediction (สีแดง)
        plt.plot(test_final['record_date'], test_final['pred_price'], label='Test (Exam)', color='red', alpha=0.9, linewidth=1.5)
        
        # เส้นแบ่งช่วง
        train_end_date = train_final['record_date'].max()
        val_end_date = val_final['record_date'].max()
        
        plt.axvline(train_end_date, color='gray', linestyle='--', label='Train End')
        plt.axvline(val_end_date, color='gray', linestyle='--', label='Val End')
        
        plt.title(f"XGBoost Model Performance (Test Acc: {test_acc*100:.2f}%, MAPE: {test_mape*100:.2f}%)")
        plt.xlabel("Date")
        plt.ylabel("THB/USD")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        print("[PLOT] Displaying Full Prediction Graph... (Close window to finish)")
        plt.show() 
        
        # Feature Importance
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(model, ax=ax, max_num_features=12, height=0.5, title="Top Factors Driving THB/USD")
        plt.tight_layout()
        print("[PLOT] Displaying Importance Graph... (Close window to finish)")
        plt.show()
        
    except Exception as e:
        print(f"Plot Error: {e}")

if __name__ == "__main__":
    train()