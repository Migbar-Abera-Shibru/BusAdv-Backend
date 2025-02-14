from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = 'Users'
    UserID = Column(Integer, primary_key=True, autoincrement=True)
    Username = Column(String(255), nullable=False)
    Email = Column(String(255), nullable=False, unique=True)
    Role = Column(String(255), nullable=False)
    Preferences = Column(Text)
    CreatedAt = Column(DateTime, default=datetime.utcnow)
    LastLogin = Column(DateTime, default=None)
    IsActive = Column(Boolean, default=True)

    # Define a relationship to the Project model using a string to avoid circular imports
    projects = relationship("Project", back_populates="owner")