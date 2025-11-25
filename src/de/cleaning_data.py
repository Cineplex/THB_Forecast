import pandas as pd
from src.de.utils.db_connector import get_engine

def clean_and_load_cleaning_data():
    engine = get_engine()
    if not engine: return

    print("⏳ [Cleaning_data] Processing & Cleaning Data...")

    # 1. โหลดข้อมูลดิบจาก Bronze
    df_raw = pd.read_sql("SELECT * FROM raw_data ORDER BY record_date", engine)
    df_raw['record_date'] = pd.to_datetime(df_raw['record_date'])
    df_raw = df_raw.set_index('record_date')

    # 2. Cleaning: Forward Fill (เติมข้อมูลวันหยุดด้วยราคาล่าสุด)
    # นี่คือหัวใจสำคัญของการทำ Silver Layer สำหรับ Time Series
    df_clean = df_raw.fillna(method='ffill')
    
    # ลบแถวแรกๆ ที่ยังเป็น NaN (ก่อนวันแรกที่มีข้อมูล)
    df_clean = df_clean.dropna()

    # 3. Rename Columns ให้สวยงาม (ตัด _raw, _close ออก)
    rename_map = {
        'thb_usd_close': 'thb_usd',
        'gold_close': 'gold',
        'oil_close': 'oil',
        'bond_yield_raw': 'bond_yield',
        'dxy_raw': 'dxy',
        'sp500_raw': 'sp500',
        'set_index_raw': 'set_index'
    }
    df_clean = df_clean.rename(columns=rename_map)

    # 4. Save to Cleaning Data Table
    df_clean = df_clean.reset_index()
    df_clean.to_sql('cleaning_data', engine, if_exists='replace', index=False)
    print(f"✅ Clean Data Saved! ({len(df_clean)} rows)")

if __name__ == "__main__":
    clean_and_load_cleaning_data()