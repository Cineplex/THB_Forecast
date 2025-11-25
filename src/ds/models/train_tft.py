import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from src.de.utils.db_connector import get_engine
import warnings

warnings.filterwarnings("ignore")

# --- Config (Deep Tuning) ---
MAX_ENCODER_LENGTH = 120
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

# --- 1. Load Data (From Feature Layer) ---
def load_feature_data():
    engine = get_engine()
    if not engine: return None
    print("📥 Loading data from Feature Layer (feature_data)...")
    
    # ดึงข้อมูลทั้งหมด (Feature คำนวณและ Shift มาแล้ว)
    query = "SELECT * FROM feature_data ORDER BY record_date ASC"
    
    df = pd.read_sql(query, engine)
    df['record_date'] = pd.to_datetime(df['record_date'])
    
    # Clean Duplicates
    df = df.drop_duplicates(subset=['record_date'], keep='last')
    
    return df

def train():
    # Load
    df = load_feature_data()
    if df is None: return

    # --- 2. Prepare Data ---
    # ไม่ต้องคำนวณ Feature แล้ว แค่เลือกคอลัมน์
    feature_cols = [
        'gold', 'oil', 'bond_yield', 'dxy', 'sp500', 'set_index', 
        'rsi', 'macd', 'pct_change',
        'volatility_5', 'volatility_20',
        'gold_oil_ratio', 'bond_dxy_ratio', 'dist_sma20'
    ]
    # กรองเฉพาะที่มีจริง
    feature_cols = [c for c in feature_cols if c in df.columns]

    # TFT Setup
    df = df.reset_index() if 'record_date' not in df.columns else df
    df['time_idx'] = (df['record_date'] - df['record_date'].min()).dt.days
    df['group_id'] = 'THB_USD'
    
    # Convert Categories
    cat_cols = ['month', 'day_of_week', 'is_holiday_th']
    for c in cat_cols:
        if c in df.columns: df[c] = df[c].astype(str)
        
    # --- 3. Split Data ---
    val_start_idx = df[df['record_date'] >= "2023-01-01"].iloc[0]['time_idx']
    test_start_idx = df[df['record_date'] >= "2024-01-01"].iloc[0]['time_idx']
    
    training_cutoff = val_start_idx - 1
    validation_cutoff = test_start_idx - 1

    print("-" * 50)
    print(f"📌 Split Info: Train End={training_cutoff}, Val End={validation_cutoff}")
    print("-" * 50)

    # --- 4. Create Datasets ---
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_diff",
        group_ids=["group_id"],
        min_encoder_length=60,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        allow_missing_timesteps=True,
        
        # Features ที่ Shift มาแล้ว ถือเป็น Unknown Reals (อดีต)
        time_varying_unknown_reals=["target_diff"] + feature_cols,
        time_varying_known_categoricals=cat_cols,
        
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in cat_cols},
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        df[lambda x: x.time_idx <= validation_cutoff], 
        predict=False, stop_randomization=True,
        min_prediction_idx=training_cutoff + 1
    )
    
    testing = TimeSeriesDataSet.from_dataset(
        training, 
        df, 
        predict=False, stop_randomization=True,
        min_prediction_idx=validation_cutoff + 1
    )
    
    train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE*10, num_workers=0)
    test_dl = testing.to_dataloader(train=False, batch_size=BATCH_SIZE*10, num_workers=0)

    # --- 5. Model Setup ---
    print("🧠 Building TFT Model...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=48,         
        attention_head_size=4,
        dropout=0.45,           
        hidden_continuous_size=16,
        optimizer="adam",
        weight_decay=1e-3,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=5,
    )

    # --- 6. Train ---
    # Check GPU
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training TFT on {accelerator.upper()} ({EPOCHS} Epochs)...")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator, devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[
            LearningRateMonitor(), 
            EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-5) 
        ],
        logger=True
    )
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # --- 7. Evaluate ---
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"🏆 Loading Best Model: {best_model_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    def calculate_metrics(dataloader):
        raw_preds = best_tft.predict(dataloader, mode="raw", return_x=True)
        pred_diff = raw_preds.output.prediction[:, 0, 3].cpu().numpy() # Median
        time_idx = raw_preds.x['decoder_time_idx'][:, 0].cpu().numpy()
        
        res = pd.DataFrame({'time_idx': time_idx, 'pred_diff': pred_diff})
        final = pd.merge(res, df, on='time_idx', how='left')
        
        # Reconstruct: Price(t) = Price(t-1) + Diff(t)
        # ใน Feature Layer มี lag_1 อยู่แล้ว
        final['pred_price'] = final['lag_1'] + final['pred_diff']
        
        # Actual Price: ต้องดึง thb_usd มาเทียบ (หรือคำนวณจาก lag_1 + target_diff)
        # ใช้ lag_1 + target_diff ดีกว่าเพราะมันอยู่ใน Feature Layer แน่ๆ
        final['actual_price'] = final['lag_1'] + final['target_diff']
        
        mape = mean_absolute_percentage_error(final['actual_price'], final['pred_price'])
        return mape, final

    print("\n🔍 Check Overfitting...")
    train_eval_dl = training.to_dataloader(train=False, batch_size=BATCH_SIZE*10, num_workers=0)
    train_mape, train_final = calculate_metrics(train_eval_dl)
    val_mape, val_final = calculate_metrics(val_dl)
    test_mape, test_final = calculate_metrics(test_dl)

    print("-" * 50)
    print(f"🏆 Overfitting Check Results:")
    print(f"   Train MAPE: {train_mape * 100:.4f}%")
    print(f"   Val   MAPE: {val_mape * 100:.4f}%")
    print(f"   Test  MAPE: {test_mape * 100:.4f}%")
    print(f"   Gap       : {(test_mape - train_mape) * 100:.4f}%")
    
    # Final Metrics
    final = test_final
    threshold = 0.005 
    meaningful_change = (final['target_diff'].abs() > threshold) & (final['pred_diff'].abs() > threshold)
    
    dir_acc = 0.0
    if meaningful_change.sum() > 0:
        dir_acc = (np.sign(final.loc[meaningful_change, 'target_diff']) == np.sign(final.loc[meaningful_change, 'pred_diff'])).mean()
    
    raw_dir_acc = (np.sign(final['target_diff']) == np.sign(final['pred_diff'])).mean()
    mae = mean_absolute_error(final['actual_price'], final['pred_price'])
    
    print("-" * 50)
    print(f"🎯 TFT Final Performance :")
    print(f"   Raw Accuracy: {raw_dir_acc * 100:.2f}%")
    print(f"   Significant Acc: {dir_acc * 100:.2f}%")
    print(f"   Error (MAE): {mae:.4f} THB")
    print("-" * 50)
    
    # --- Save Model ---
    try:
        import os
        import shutil
        import joblib
        from datetime import datetime
        
        # สร้างโฟลเดอร์ save_models ถ้ายังไม่มี
        model_dir = os.path.join("src", "ds", "models", "save_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy checkpoint ไว้ในโฟลเดอร์เดียวกัน
        checkpoint_filename = "tft_checkpoint.ckpt"
        checkpoint_path = os.path.join(model_dir, checkpoint_filename)
        shutil.copy2(best_model_path, checkpoint_path)
        
        # รวมทุกอย่างในไฟล์เดียว
        model_package = {
            'checkpoint_path': checkpoint_filename,  # relative path
            'features': feature_cols,
            'categorical_features': cat_cols,
            'metadata': {
                'model_type': 'TFT',
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_mape': float(test_mape),
                'test_mae': float(mae),
                'test_accuracy': float(raw_dir_acc),
                'val_start_date': "2023-01-01",
                'test_start_date': "2024-01-01",
                'max_encoder_length': MAX_ENCODER_LENGTH,
                'max_prediction_length': MAX_PREDICTION_LENGTH,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS
            }
        }
        
        # บันทึกเป็นไฟล์เดียว
        model_path = os.path.join(model_dir, "tft_model.pkl")
        joblib.dump(model_package, model_path)
        
        print(f"\n💾 Model Saved Successfully!")
        print(f"   Path: {model_path}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Includes: checkpoint info + features + metadata")
        
    except Exception as e:
        print(f"\n❌ Model Save Error: {e}")
    
    try:
        # Plotting Logic
        plt.figure(figsize=(18, 8))
        
        # 1. Actual Price (เส้นจริง)
        plt.plot(df['record_date'], df['thb_usd'], label='Actual Price', color='black', alpha=0.3, linewidth=1)
        
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
        
        plt.title(f"TFT Model Performance (Test Acc: {raw_dir_acc*100:.2f}%, MAPE: {test_mape*100:.2f}%)")
        plt.xlabel("Date")
        plt.ylabel("THB/USD")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        print("[PLOT] Displaying Full Prediction Graph... (Close window to finish)")
        plt.show()
        
        # Importance
        raw_predictions = best_tft.predict(test_dl, mode="raw", return_x=True)
        interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
        best_tft.plot_interpretation(interpretation)
        print("[PLOT] Displaying Interpretation Graph... (Close window to finish)")
        plt.show()
    except Exception as e:
        print(f"Plot Error: {e}")

if __name__ == "__main__":
    train()