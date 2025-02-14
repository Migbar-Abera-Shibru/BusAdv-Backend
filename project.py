from database import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

class Project(Base):
    __tablename__ = 'projects'
    ProjectID = Column(Integer, primary_key=True, autoincrement=True)
    ProjectName = Column(String(255), nullable=False)
    OwnerID = Column(Integer, ForeignKey('Users.UserID'), nullable=False)  # Foreign key to Users table

    # Define a relationship to the User model
    owner = relationship("User", back_populates="projects")