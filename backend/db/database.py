import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Create the db directory if it doesn't exist
db_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

SQLALCHEMY_DATABASE_URL = "sqlite:///./trading.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
