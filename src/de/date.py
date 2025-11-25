import pandas as pd
import holidays
from src.de.utils.db_connector import get_engine

def load_calendar_date():
    print("📅  Generating Calendar Data...")
    
    # 1. เจนเผื่ออนาคตไปถึงปี 2030 (เผื่อไว้สำหรับ TFT หรือการทำนายระยะยาว)
    dates = pd.date_range(start="2015-01-01", end="2030-12-31")
    
    th_holidays = holidays.Thailand()
    us_holidays = holidays.US()
    
    data = []
    for dt in dates:
        data.append({
            'record_date': dt.date(),  # <--- [FIXED] เปลี่ยนชื่อให้ตรงกับตารางอื่น
            'day_of_week': dt.dayofweek, # 0=Mon, 6=Sun
            'month': dt.month,
            'day_of_month': dt.day,
            'year': dt.year,           # เพิ่มปีไว้ด้วย เผื่อใช้กรอง
            'is_holiday_th': 1 if dt in th_holidays else 0,
            'is_holiday_us': 1 if dt in us_holidays else 0,
            'is_weekend': 1 if dt.dayofweek >= 5 else 0
        })
    
    df_cal = pd.DataFrame(data)
    
    engine = get_engine()
    if engine:
        # เก็บลงตาราง fx_calendar_dim
        # ใช้ if_exists='replace' เพราะเราสร้างใหม่ทับของเดิมได้เลย ไม่ต้อง append
        df_cal.to_sql('calendar_date', engine, if_exists='replace', index=False)
        print(f"✅ Calendar Data Saved! ({len(df_cal)} rows)")

if __name__ == "__main__":
    load_calendar_date()