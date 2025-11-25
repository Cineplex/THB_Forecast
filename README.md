# THB/USD Exchange Rate Forecasting

โปรเจคพยากรณ์อัตราแลกเปลี่ยน THB/USD โดยใช้ Machine Learning และ Deep Learning

## โครงสร้างโปรเจค

```
THB_Forecast/
├── src/
│   ├── de/                          # Data Engineering
│   │   ├── raw_data.py              # ดึงข้อมูลดิบ
│   │   ├── cleaning_data.py         # ทำความสะอาดข้อมูล
│   │   ├── feature_data.py          # สร้าง features สำหรับ ML
│   │   ├── date.py                  # สร้างข้อมูลปฏิทิน
│   │   └── utils/                   # Utilities สำหรับ DE
│   │       └── db_connector.py      # เชื่อมต่อฐานข้อมูล
│   ├── ds/                          # Data Science
│   │   └── models/
│   │       ├── train_xgboost.py    # โมเดล XGBoost
│   │       ├── train_lstm.py       # โมเดล LSTM
│   │       └── train_tft.py        # โมเดล TFT
│   ├── notebook/
│   │   ├── feature_engineering.ipynb   # Notebook สำหรับ feature engineering
│   │   └── modeling_comparison.ipynb   # Notebook เปรียบเทียบโมเดล
│   ├── config.py                    # การตั้งค่าฐานข้อมูล
│   └── main.py                      # Pipeline orchestrator
├── .env                             # ตัวแปรสภาพแวดล้อม (Database config)
├── requirements.txt                 # Python dependencies
└── README.md
```

## โครงสร้างโปรเจค

### 1. Data Engineering (de/)

#### ขั้นตอนที่ 1: Data Ingestion (`raw_data.py`)
- **ดึงข้อมูลจาก**: yfinance (Yahoo Finance API)
- **ข้อมูลที่ดึง**:
  - `THB=X`: อัตราแลกเปลี่ยน THB/USD
  - `GC=F`: ราคาทอง (Gold)
  - `BZ=F`: ราคาน้ำมัน (Brent Oil)
  - `^TNX`: ผลตอบแทนพันธบัตร 10 ปี (10-Year Treasury Yield)
  - `DX-Y.NYB`: ดัชนีดอลลาร์ (DXY)
  - `^GSPC`: ดัชนี S&P 500
  - `^SET.BK`: ดัชนีตลาดหลักทรัพย์ไทย (SET Index)
- **ระบบ Incremental Load**: ดึงเฉพาะข้อมูลใหม่ที่ยังไม่มีในฐานข้อมูล
- **เก็บลง**: `raw_data`

```bash
python -m src.de.raw_data
```

#### ขั้นตอนที่ 2: Data Cleaning (`cleaning_data.py`)
- จัดการค่าที่หายไป (Forward Fill)
- ลบข้อมูลซ้ำ
- ปรับชื่อคอลัมน์ให้สวยงาม (ตัด _raw, _close ออก)
- **เก็บลง**: `cleaning_data`

```bash
python -m src.de.cleaning_data
```

#### ขั้นตอนที่ 3: Feature Engineering (`feature_data.py`)
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - SMA 50 (Simple Moving Average 50 วัน)
  - Distance from SMA 20
- **Lag Features**: lag_1, lag_7 (ข้อมูลย้อนหลัง 1, 7 วัน)
- **Rolling Statistics**: 
  - Volatility (5, 20 วัน)
- **Relationship Features**: 
  - Gold/Oil Ratio
  - Bond Yield/DXY Ratio
- **Target Variable**: target_diff (การเปลี่ยนแปลงราคา)
- **Calendar Features**: รวมข้อมูลจาก `calendar_date`
- **Feature Shifting**: Shift features 1 วันเพื่อป้องกัน Data Leakage
- **เก็บลง**: `feature_data`

```bash
python -m src.de.feature_data
```

### 2. Data Science (ds/)

#### Models

##### XGBoost (`train_xgboost.py`)
- **ประเภท**: Gradient Boosting Model
- **จุดเด่น**: 
  - ดีสำหรับข้อมูลแบบ tabular
  - Train เร็ว, รองรับ GPU
  - มี Feature Importance
- **การแบ่งข้อมูล**:
  - Train: ก่อน 2023-01-01
  - Validation: 2023-01-01 ถึง 2023-12-31
  - Test: 2024-01-01 เป็นต้นไป
- **Metrics**: MAPE, MAE, Direction Accuracy

##### LSTM (`train_lstm.py`)
- **ประเภท**: Recurrent Neural Network
- **จุดเด่น**:
  - เหมาะสำหรับข้อมูล Time Series
  - จับ Temporal Patterns ได้ดี
- **การแบ่งข้อมูล**: เหมือน XGBoost
- **Sequence Length**: 30 วัน

##### TFT (`train_tft.py`)
- **ประเภท**: Temporal Fusion Transformer
- **จุดเด่น**:
  - State-of-the-art สำหรับ Time Series Forecasting
  - รองรับ Attention Mechanism
  - จัดการ Multi-horizon Forecasting
- **การแบ่งข้อมูล**: เหมือน XGBoost
- **Max Encoder/Prediction Length**: 30 วัน

#### Notebooks

##### `feature_engineering.ipynb`
- แสดงกระบวนการ Data Engineering ทั้ง 3 ขั้นตอน:
  1. Data Ingestion
  2. Data Cleaning
  3. Feature Engineering
- วิเคราะห์คุณภาพข้อมูล
- แสดง Feature Distributions
- Correlation Analysis
- ตรวจสอบข้อมูลในแต่ละ Table

##### `modeling_comparison.ipynb`
- เปรียบเทียบโมเดลทั้ง 3 แบบ (XGBoost, LSTM, TFT)
- แสดง Features ที่ใช้ Train
- แสดง Performance Metrics:
  - MAPE (Mean Absolute Percentage Error)
  - MAE (Mean Absolute Error)
  - Direction Accuracy
- แสดงกราฟ Predictions vs Actual สำหรับ Train, Validation, Test
- Feature Importance Analysis

## การติดตั้ง

### 1. สร้าง Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. ตั้งค่าฐานข้อมูล
สร้างไฟล์ `.env` ที่ root directory:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=thb_forecasting
DB_USER=postgres
DB_PASS=your_password
```

สร้างฐานข้อมูลใน PostgreSQL:
```sql
CREATE DATABASE thb_forecasting;
```

## การใช้งาน

### วิธีที่ 1: ใช้ main.py (แนะนำ)

```bash
# รัน Data Engineering Pipeline ทั้งหมด
python -m src.main -de

# รัน Data Science (Model Training) Pipeline ทั้งหมด
python -m src.main -ds

# รันทั้ง Data Engineering และ Model Training
python -m src.main -all
```

### วิธีที่ 2: รันแต่ละไฟล์

```bash
# Data Engineering
python -m src.de.raw_data          # ดึงข้อมูลดิบ
python -m src.de.date              # สร้างข้อมูลปฏิทิน
python -m src.de.cleaning_data     # ทำความสะอาด
python -m src.de.feature_data      # สร้าง features

# Data Science
python -m src.ds.models.train_xgboost
python -m src.ds.models.train_lstm
python -m src.ds.models.train_tft
```

### วิธีที่ 3: ใช้ Jupyter Notebook

```bash
jupyter notebook
# เปิดไฟล์ใน src/notebook/
```

## Database Tables

| Table Name | Description | Source | Key Columns |
|------------|-------------|--------|-------------|
| `raw_data` | ข้อมูลตลาดดิบจาก Yahoo Finance | yfinance | record_date, thb_usd_close, gold_close, oil_close, bond_yield_raw, dxy_raw, sp500_raw, set_index_raw |
| `calendar_date` | ข้อมูลปฏิทิน (วันหยุด, วันธรรมดา) | สร้างจาก date.py | record_date, day_of_week, month, is_holiday_th, is_holiday_us, is_weekend |
| `cleaning_data` | ข้อมูลตลาดที่ทำความสะอาดแล้ว | raw_data + cleaning | record_date, thb_usd, gold, oil, bond_yield, dxy, sp500, set_index |
| `feature_data` | ข้อมูล Features สำหรับ Machine Learning | cleaning_data + calendar_date + feature engineering | ทุกคอลัมน์จาก cleaning_data + technical indicators + lag features + calendar features + target_diff |

## Features ที่สร้าง

### Technical Indicators
- **RSI**: Relative Strength Index (14 วัน)
- **MACD**: Moving Average Convergence Divergence
- **SMA 50**: Simple Moving Average 50 วัน
- **Distance from SMA 20**: ระยะห่างจาก SMA 20 (เปอร์เซ็นต์)
- **PCT Change**: Percentage Change

### Volatility Features
- **volatility_5**: ความผันผวน 5 วัน (Rolling Standard Deviation)
- **volatility_20**: ความผันผวน 20 วัน (Rolling Standard Deviation)

### Lag Features
- **lag_1**: ราคา THB/USD เมื่อ 1 วันก่อน
- **lag_7**: ราคา THB/USD เมื่อ 7 วันก่อน

### Relationship Features
- **gold_oil_ratio**: อัตราส่วน Gold/Oil
- **bond_dxy_ratio**: อัตราส่วน Bond Yield/DXY

### Calendar Features (จาก calendar_date)
- **day_of_week**: วันในสัปดาห์
- **month**: เดือน
- **is_holiday_th**: วันหยุดไทย
- **is_holiday_us**: วันหยุดสหรัฐ

### Target Variable
- **target_diff**: การเปลี่ยนแปลงของราคา THB/USD (Price(t) - Price(t-1))

**รวม**: ~20 features + original market data columns

## Model Performance

| Model | MAPE (Test) | MAE (Test) | Direction Accuracy (Test) |
|-------|-------------|------------|---------------------------|
| XGBoost | อัปเดตเมื่อรัน | อัปเดตเมื่อรัน | อัปเดตเมื่อรัน |
| LSTM | อัปเดตเมื่อรัน | อัปเดตเมื่อรัน | อัปเดตเมื่อรัน |
| TFT | อัปเดตเมื่อรัน | อัปเดตเมื่อรัน | อัปเดตเมื่อรัน |

*หมายเหตุ: รัน `modeling_comparison.ipynb` หรือ train แต่ละโมเดลเพื่อดู Performance ล่าสุด*

**การอ่านผลลัพธ์**:
- **MAPE (Mean Absolute Percentage Error)**: ค่าเฉลี่ยของเปอร์เซ็นต์ความคลาดเคลื่อน (ยิ่งต่ำยิ่งดี)
- **MAE (Mean Absolute Error)**: ค่าเฉลี่ยของความคลาดเคลื่อนในหน่วย THB (ยิ่งต่ำยิ่งดี)
- **Direction Accuracy**: ความแม่นยำในการทำนายทิศทาง (ขึ้น/ลง) (ยิ่งสูงยิ่งดี)

## Requirements

### ซอฟต์แวร์
- Python 3.11+
- PostgreSQL 12+
- CUDA (optional, สำหรับ GPU training ของ XGBoost และ PyTorch)

### Python Packages
```
pandas
numpy
yfinance
sqlalchemy
psycopg2-binary
python-dotenv
scikit-learn
xgboost
torch
pytorch-forecasting
matplotlib
holidays
lightning
ta
```

## ผู้พัฒนา

**โครงสร้างทีม**:
- **Data Engineering**: ดึงข้อมูลดิบ, ทำความสะอาด, สร้าง Features
- **Data Science**: พัฒนาและ Train โมเดล Machine Learning

**สถาบัน**: มหาวิทยาลัย CSS (Computer Science & Statistics)
**หลักสูตร**: Data Science & Data Engineer

## License

MIT License
