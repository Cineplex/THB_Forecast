import pandas as pd
import ta
import numpy as np
from src.de.utils.db_connector import get_engine

def build_feature_data():
    engine = get_engine()
    if not engine: return

    print("🧠 [Feature Data] Starting Feature Engineering...")

    # --- 1. Load Silver Data ---
    try:
        df = pd.read_sql("SELECT * FROM cleaning_data ORDER BY record_date", engine)
        df['record_date'] = pd.to_datetime(df['record_date'])
    except Exception as e:
        print(f"❌ Error loading Silver data: {e}")
        return

    # --- 2. Load Calendar Data & Merge ---
    try:
        df_cal = pd.read_sql("SELECT * FROM calendar_date", engine)
        df_cal['record_date'] = pd.to_datetime(df_cal['record_date'])
        
        df = pd.merge(df, df_cal, on='record_date', how='left')
        print(f"   -> Merged with Calendar ({len(df)} rows)")
    except Exception as e:
        print(f"⚠️ Warning: Could not load Calendar table ({e})")

    df = df.set_index('record_date')

    # --- 3. Feature Engineering ---
    
    # 3.1 Volatility
    df['volatility_5'] = df['thb_usd'].rolling(window=5).std()
    df['volatility_20'] = df['thb_usd'].rolling(window=20).std()
    
    # 3.2 Ratios
    df['gold_oil_ratio'] = df['gold'] / df['oil']
    df['bond_dxy_ratio'] = df['bond_yield'] / df['dxy']
    
    # 3.3 Technical Indicators
    # [FIXED] บันทึก sma_50 ลง DataFrame โดยตรง (เพราะต้องใช้ Shift ทีหลัง)
    df['sma_50'] = ta.trend.sma_indicator(df['thb_usd'], window=50)
    
    # คำนวณ Distance โดยใช้คอลัมน์ที่เพิ่งสร้าง
    df['dist_sma20'] = (df['thb_usd'] - df['sma_50']) / df['sma_50']
    
    df['rsi'] = ta.momentum.rsi(df['thb_usd'], window=14)
    df['macd'] = ta.trend.macd(df['thb_usd'])
    df['pct_change'] = df['thb_usd'].pct_change()

    # 3.4 Target
    df['target_diff'] = df['thb_usd'].diff()
    
    # 3.5 Lag Features
    df['lag_1'] = df['thb_usd'].shift(1)
    df['lag_7'] = df['thb_usd'].shift(7)

    # --- 4. Shift Features ---
    features_to_shift = [
        'gold', 'oil', 'bond_yield', 'dxy', 'sp500', 'set_index',
        'rsi', 'macd', 'pct_change', 'sma_50',
        'volatility_5', 'volatility_20',
        'gold_oil_ratio', 'bond_dxy_ratio', 'dist_sma20'
    ]
    
    for col in features_to_shift:
        # เช็คก่อนเผื่อ column ไหนไม่มีจริง (เช่น คำนวณแล้ว error)
        if col in df.columns:
            df[col] = df[col].shift(1)
        else:
            print(f"⚠️ Warning: Feature '{col}' not found in DataFrame. Skipping shift.")

    # Clean Data
    df = df.dropna()
    
    # จัดการ Data Type ของปฏิทิน
    cat_cols = ['month', 'day_of_week', 'is_holiday_th', 'is_holiday_us']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    
    if 'is_weekend' in df.columns:
        df = df.drop(columns=['is_weekend'])

    # --- 5. Save to Feature Table ---
    df = df.reset_index()
    try:
        df.to_sql('feature_data', engine, if_exists='replace', index=False)
        print(f"✅ [Feature] Features Built & Saved! ({len(df)} rows) -> Table: feature_data")
    except Exception as e:
        print(f"❌ Error saving to Feature table: {e}")

if __name__ == "__main__":
    build_feature_data()