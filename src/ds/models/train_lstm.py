import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.de.utils.db_connector import get_engine

# --- Config ---
SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 300
LEARNING_RATE = 0.001
VAL_START_DATE = "2023-01-01"
TEST_START_DATE = "2024-01-01"

# --- Load Data from Feature Layer Directly ---
def load_feature_data():
    engine = get_engine()
    if not engine: return None
    print("📥 Loading data from Feature Layer (feature_data)...")
    
    # ดึงข้อมูลทั้งหมดจากตาราง Feature
    query = "SELECT * FROM feature_data ORDER BY record_date ASC"
    
    try:
        df = pd.read_sql(query, engine)
        df['record_date'] = pd.to_datetime(df['record_date'])
        # ลบข้อมูลซ้ำ (ถ้ามี)
        df = df.drop_duplicates(subset=['record_date'], keep='last')
        df = df.set_index('record_date')
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(input_data, target_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i + seq_length)]
        y = target_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train():
    # 1. Load Data
    df = load_feature_data()
    if df is None: return

    # 2. Select Features (ที่มีอยู่ใน Feature Table)
    feature_cols = [
        'gold', 'oil', 'bond_yield', 'dxy', 'sp500', 'set_index', 
        'rsi', 'macd', 'pct_change',
        'volatility_5', 'volatility_20',
        'gold_oil_ratio', 'bond_dxy_ratio', 'dist_sma20',
        'lag_1', 'lag_7'
    ]
    # กรองเฉพาะที่มีจริง
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    target_col = 'target_diff'
    
    # 3. Scaling (จำเป็นสำหรับ LSTM)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 4. Split Data (Train / Val / Test)
    dates = df.index
    
    # Train+Val Mask (สำหรับเทรน)
    train_val_mask = dates < TEST_START_DATE
    
    # Test Mask (สำหรับสอบ)
    test_mask = dates >= TEST_START_DATE
    
    # Train Mask เฉยๆ (สำหรับเช็ค Overfit)
    train_only_mask = dates < VAL_START_DATE
    
    # --- [FIXED] Data Leakage Prevention ---
    # Fit Scaler เฉพาะ Train Set เท่านั้น
    X_train_val_raw = df.loc[train_val_mask, feature_cols].values
    y_train_val_raw = df.loc[train_val_mask, [target_col]].values
    
    scaler_x.fit(X_train_val_raw)
    scaler_y.fit(y_train_val_raw)
    
    # Transform ข้อมูลทั้งหมด
    scaled_x_all = scaler_x.transform(df[feature_cols].values)
    scaled_y_all = scaler_y.transform(df[[target_col]].values)
    
    # --- [FIXED] Sequence Gap Handling ---
    # เตรียมข้อมูลสำหรับ Test โดยต้องรวม Lookback (60 วันก่อนหน้า)
    # หา index ของวันแรกที่จะ Test
    test_start_idx = np.where(test_mask)[0][0]
    
    # ถอยหลังไป SEQUENCE_LENGTH วัน เพื่อให้ทำนายวันแรกของ Test ได้
    lookback_start_idx = max(0, test_start_idx - SEQUENCE_LENGTH)
    
    # ข้อมูลสำหรับสร้าง Test Sequence (รวม Lookback แล้ว)
    X_test_with_lookback = scaled_x_all[lookback_start_idx:]
    y_test_with_lookback = scaled_y_all[lookback_start_idx:]
    
    # ข้อมูล Train (ใช้แบบเดิมได้เลยเพราะเริ่มจาก 0)
    X_train_val = scaled_x_all[train_val_mask]
    y_train_val = scaled_y_all[train_val_mask]
    
    print("-" * 50)
    print(f"📌 Split Info:")
    print(f"   Train+Val Set: {len(X_train_val)} rows")
    print(f"   Test Set (Raw): {np.sum(test_mask)} rows")
    print("-" * 50)
    
    # 5. Create Sequences
    # Train Sequence
    X_train, y_train = create_sequences(X_train_val, y_train_val, SEQUENCE_LENGTH)
    
    # Test Sequence (สร้างจากข้อมูลที่มี Lookback)
    X_test, y_test = create_sequences(X_test_with_lookback, y_test_with_lookback, SEQUENCE_LENGTH)
    
    # Check Sequence (สำหรับวัดผล Train กลับ)
    train_only_indices = np.where(train_only_mask)[0]
    X_chk_raw = scaled_x_all[train_only_indices]
    y_chk_raw = scaled_y_all[train_only_indices]
    X_chk, y_chk = create_sequences(X_chk_raw, y_chk_raw, SEQUENCE_LENGTH)
    
    # Convert to Tensor
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available. Training on CPU.")
        print("   (To enable GPU, please install PyTorch with CUDA support: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)")

    print(f"🚀 Training on {device}")
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    X_chk_t = torch.from_numpy(X_chk).float().to(device)
    
    # 6. Model Setup
    model = LSTMModel(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    print(f"🚀 Training LSTM ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if (epoch+1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

    # 7. Evaluation
    model.eval()
    with torch.no_grad():
        # --- A. Predict Train (Check Overfit) ---
        chk_pred_scaled = model(X_chk_t).cpu().numpy()
        chk_pred_diff = scaler_y.inverse_transform(chk_pred_scaled)
        
        # Reconstruct Train Price
        chk_indices = df[train_only_mask].index[SEQUENCE_LENGTH:]
        chk_base_price = df.loc[chk_indices, 'lag_1'].values.reshape(-1, 1)
        chk_actual_price = df.loc[chk_indices, 'thb_usd'].values.reshape(-1, 1)
        chk_pred_price = chk_base_price + chk_pred_diff
        
        train_mape = mean_absolute_percentage_error(chk_actual_price, chk_pred_price)
        
        # --- B. Predict Validation ---
        val_mask = (dates >= VAL_START_DATE) & (dates < TEST_START_DATE)
        val_indices_all = np.where(val_mask)[0]
        
        if len(val_indices_all) > SEQUENCE_LENGTH:
            # สร้าง Validation Sequences
            val_start_idx = max(0, val_indices_all[0] - SEQUENCE_LENGTH)
            X_val_with_lookback = scaled_x_all[val_start_idx:]
            y_val_with_lookback = scaled_y_all[val_start_idx:]
            
            # สร้าง Sequences
            X_val_seq, y_val_seq = create_sequences(X_val_with_lookback, y_val_with_lookback, SEQUENCE_LENGTH)
            
            # ตัดให้เหลือเฉพาะส่วนที่เป็น val period
            num_val_samples = len(val_indices_all)
            X_val_seq = X_val_seq[:num_val_samples]
            
            X_val_t = torch.from_numpy(X_val_seq).float().to(device)
            val_pred_scaled = model(X_val_t).cpu().numpy()
            val_pred_diff = scaler_y.inverse_transform(val_pred_scaled)
            
            # Reconstruct Val Price
            val_indices = df[val_mask].index[:len(val_pred_diff)]
            val_base_price = df.loc[val_indices, 'lag_1'].values.reshape(-1, 1)
            val_actual_price = df.loc[val_indices, 'thb_usd'].values.reshape(-1, 1)
            val_pred_price = val_base_price + val_pred_diff
            
            val_mape = mean_absolute_percentage_error(val_actual_price, val_pred_price)
        else:
            val_pred_price = np.array([])
            val_indices = pd.DatetimeIndex([])
            val_mape = 0.0
        
        # --- C. Predict Test ---
        pred_diff_scaled = model(X_test_t).cpu().numpy()
        pred_diff = scaler_y.inverse_transform(pred_diff_scaled)
        
        # Reconstruct Price (Test)
        test_indices = df[test_mask].index[:len(pred_diff)]
        
        base_price = df.loc[test_indices, 'lag_1'].values.reshape(-1, 1)
        actual_price = df.loc[test_indices, 'thb_usd'].values.reshape(-1, 1)
        
        pred_price = base_price + pred_diff
        
        test_mae = mean_absolute_error(actual_price, pred_price)
        test_mape = mean_absolute_percentage_error(actual_price, pred_price)
        
        # Direction Accuracy
        actual_diff = df.loc[test_indices, 'target_diff'].values.reshape(-1, 1)
        
        # Significant Move (> 0.005)
        threshold = 0.005
        sig_idx = (np.abs(actual_diff) > threshold) & (np.abs(pred_diff) > threshold)
        if np.sum(sig_idx) > 0:
            sig_acc = (np.sign(pred_diff[sig_idx]) == np.sign(actual_diff[sig_idx])).mean()
        else: sig_acc = 0.0
            
        raw_acc = (np.sign(pred_diff) == np.sign(actual_diff)).mean()

        print("-" * 50)
        print(f"🏆 Overfitting Check Results:")
        print(f"   Train MAPE: {train_mape * 100:.4f}%")
        print(f"   Val   MAPE: {val_mape * 100:.4f}%")
        print(f"   Test  MAPE: {test_mape * 100:.4f}%")
        print(f"   Gap       : {(test_mape - train_mape) * 100:.4f}%")
        
        print("-" * 50)
        print(f"🎯 LSTM Final Performance :")
        print(f"   Raw Accuracy: {raw_acc * 100:.2f}%")
        print(f"   Significant Acc: {sig_acc * 100:.2f}%")
        print(f"   Error (MAE): {test_mae:.4f} THB")
        print("-" * 50)
        
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
                'model_state_dict': model.state_dict(),
                'model_architecture': {
                    'input_size': X_train.shape[2],
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'sequence_length': SEQUENCE_LENGTH,
                },
                'scalers': {
                    'scaler_x': scaler_x,
                    'scaler_y': scaler_y
                },
                'features': feature_cols,
                'metadata': {
                    'model_type': 'LSTM',
                    'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'test_mape': float(test_mape),
                    'test_mae': float(test_mae),
                    'test_accuracy': float(raw_acc),
                    'val_start_date': VAL_START_DATE,
                    'test_start_date': TEST_START_DATE,
                    'epochs': EPOCHS,
                    'learning_rate': LEARNING_RATE
                }
            }
            
            # บันทึกเป็นไฟล์เดียว
            model_path = os.path.join(model_dir, "lstm_model.pkl")
            joblib.dump(model_package, model_path)
            
            print(f"\n💾 Model Saved Successfully!")
            print(f"   Path: {model_path}")
            print(f"   Includes: model + scalers + features + metadata")
            
        except Exception as e:
            print(f"\n❌ Model Save Error: {e}")
        
        try:
            # Prepare DataFrames for plotting
            train_final = pd.DataFrame({'record_date': chk_indices, 'pred_price': chk_pred_price.flatten()})
            val_final = pd.DataFrame({'record_date': val_indices, 'pred_price': val_pred_price.flatten()}) if len(val_pred_price) > 0 else pd.DataFrame()
            test_final = pd.DataFrame({'record_date': test_indices, 'pred_price': pred_price.flatten()})
            
            # Reset index to make record_date a column
            df_plot = df.reset_index()

            # Plotting Logic
            plt.figure(figsize=(18, 8))
            
            # 1. Actual Price (เส้นจริง)
            plt.plot(df_plot['record_date'], df_plot['thb_usd'], label='Actual Price', color='black', alpha=0.3, linewidth=1)
            
            # 2. Train Prediction (สีเขียว)
            plt.plot(train_final['record_date'], train_final['pred_price'], label='Train (Learn)', color='green', alpha=0.8, linewidth=1)
            
            # 3. Validation Prediction (สีส้ม)
            if not val_final.empty:
                plt.plot(val_final['record_date'], val_final['pred_price'], label='Validation (Tune)', color='orange', alpha=0.9, linewidth=1.5)
            
            # 4. Test Prediction (สีแดง)
            plt.plot(test_final['record_date'], test_final['pred_price'], label='Test (Exam)', color='red', alpha=0.9, linewidth=1.5)
            
            # เส้นแบ่งช่วง
            train_end_date = train_final['record_date'].max()
            if not val_final.empty:
                val_end_date = val_final['record_date'].max()
            else:
                val_end_date = train_end_date
            
            plt.axvline(train_end_date, color='gray', linestyle='--', label='Train End')
            plt.axvline(val_end_date, color='gray', linestyle='--', label='Val End')
            
            plt.title(f"LSTM Model Performance (Test Acc: {raw_acc*100:.2f}%, MAPE: {test_mape*100:.2f}%)")
            plt.xlabel("Date")
            plt.ylabel("THB/USD")
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            print("[PLOT] Displaying Full Prediction Graph... (Close window to finish)")
            plt.show()
        except Exception as e:
            print(f"Plot Error: {e}")

if __name__ == "__main__":
    train()