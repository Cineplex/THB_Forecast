from sqlalchemy import create_engine
from src.config import DATABASE_URL

def get_engine():
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return None