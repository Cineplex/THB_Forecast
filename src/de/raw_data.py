import yfinance as yf
import pandas as pd
from sqlalchemy import text
from datetime import date, timedelta
from src.de.utils.db_connector import get_engine

# Tickers
TICKERS = {
    'THB=X': 'thb_usd_close',
    'GC=F': 'gold_close',
    'BZ=F': 'oil_close',
    '^TNX': 'bond_yield_raw',
    'DX-Y.NYB': 'dxy_raw',
    '^GSPC': 'sp500_raw',
    '^SET.BK': 'set_index_raw'
}

def get_latest_date(engine):
    """เช็ควันที่ล่าสุดที่มีอยู่ใน Database"""
    try:
        with engine.connect() as conn:
            # Query หาค่าวันที่มากที่สุด
            result = conn.execute(text("SELECT MAX(record_date) FROM raw_data"))
            latest_date = result.scalar()
            
        if latest_date:
            return pd.to_datetime(latest_date).date()
        return None
    except Exception as e:
        # กรณีที่ยังไม่มีตาราง หรือ Error อื่นๆ
        return None

def fetch_yahoo_data(start_date):
    """ดึงข้อมูลโดยระบุวันเริ่มต้น"""
    print(f"   -> Fetching Yahoo Market Data from {start_date}...")
    
    df = yf.download(list(TICKERS.keys()), start=start_date)['Close']
    
    if df.empty:
        return df
        
    df = df.rename(columns=TICKERS)
    df.index.name = 'record_date'
    df = df.reset_index()
    df['record_date'] = pd.to_datetime(df['record_date']).dt.date
    
    return df

def load_to_raw_data():
    print("🚀 Starting Raw Data Ingestion (Incremental)...")
    engine = get_engine()
    if not engine: return

    # 1. เช็ควันที่ล่าสุด
    latest_date = get_latest_date(engine)
    
    # 2. กำหนดวันที่จะดึง และ Mode การบันทึก
    if latest_date:
        # ถ้ามีข้อมูลแล้ว ให้ดึงวันถัดไป (Next Day)
        start_date = latest_date + timedelta(days=1)
        load_mode = 'append' # ต่อท้าย
        print(f"   🔄 Found existing data up to {latest_date}")
        
        # เช็คว่า start_date เกินวันนี้ไปหรือยัง
        if start_date > date.today():
            print("   ✅ Data is already up to date. No new ingestion needed.")
            return
    else:
        # ถ้ายังไม่มีข้อมูลเลย ให้ดึงใหม่หมด
        start_date = "2016-01-01"
        load_mode = 'replace' # สร้างใหม่
        print("   🆕 No existing data found. Performing Full Load.")

    # 3. ดึงข้อมูล
    df_raw = fetch_yahoo_data(start_date)
    
    if df_raw.empty:
        print("   ⚠️ No new data available from Yahoo.")
        return

    # 4. บันทึกลง DB
    try:
        # ใช้ if_exists ตามที่เรากำหนด (append หรือ replace)
        df_raw.to_sql('raw_data', engine, if_exists=load_mode, index=False)
        print(f"✅ DE RAW_DATA Success! Added {len(df_raw)} new rows (Mode: {load_mode}).")
    except Exception as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    load_to_raw_data()