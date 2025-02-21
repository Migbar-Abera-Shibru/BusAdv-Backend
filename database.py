from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Replace with your MySQL connection details
DATABASE_URL = "mysql+pymysql://root:1234@127.0.0.1:3306/my_business_advisor"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_recycle=3600)  # Avoid connection timeout

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = declarative_base()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
