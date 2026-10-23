import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Check for explicit MySQL environment variables or fallback to SQLite
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB", "voice_analysis")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL and MYSQL_HOST and MYSQL_USER:
    DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

engine = None
USE_SQLITE = False

if DATABASE_URL and DATABASE_URL.startswith("mysql"):
    try:
        # Try MySQL connection
        engine = create_engine(
            DATABASE_URL,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            pass
        logger.info(f"Connected successfully to MySQL: {DATABASE_URL}")
    except Exception as e:
        logger.warning(
            f"MySQL connection to {DATABASE_URL} failed ({e}). "
            "Falling back to local SQLite database (voice_analysis.db) so you can run out-of-the-box."
        )
        engine = None
        USE_SQLITE = True

if engine is None:
    # Safe SQLite database for instant zero-config launch (canonical backend/voice_analysis.db)
    db_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_analysis.db").replace("\\", "/")
    SQLITE_URL = f"sqlite:///{db_file_path}"
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    logger.info("Using SQLite database at ./voice_analysis.db")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
