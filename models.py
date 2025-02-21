from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base  # Import Base from database.py

# Define the User model
class User(Base):
    __tablename__ = "Users"

    UserID = Column(Integer, primary_key=True, autoincrement=True)
    Username = Column(String(255), nullable=False)
    Email = Column(String(255), nullable=False, unique=True)
    Role = Column(String(255), nullable=False)
    Preferences = Column(Text)
    CreatedAt = Column(DateTime, default=datetime.utcnow)
    LastLogin = Column(DateTime)
    IsActive = Column(Boolean, default=True)

    advices = relationship("Advice", back_populates="user")

# Define the Advice model
class Advice(Base):
    __tablename__ = "Advice"

    AdviceID = Column(Integer, primary_key=True, autoincrement=True)
    UserID = Column(Integer, ForeignKey("Users.UserID"), nullable=False)
    Context = Column(String(255), nullable=False)
    AdviceText = Column(Text, nullable=False)
    CreatedAt = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="advices")
