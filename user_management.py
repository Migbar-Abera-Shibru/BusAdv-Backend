from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from users import User
from project import Project  # Import the Project model

class UserManager:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def create_user(self, username: str, email: str, role: str, preferences: str = None) -> User:
        try:
            new_user = User(Username=username, Email=email, Role=role, Preferences=preferences)
            self.db_session.add(new_user)
            self.db_session.commit()
            return new_user
        except IntegrityError as e:
            self.db_session.rollback()
            raise ValueError(f"Duplicate email or other integrity error: {e}")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise RuntimeError(f"Database error: {e}")

    def update_user(self, user_id: int, username: str = None, email: str = None, role: str = None, preferences: str = None) -> User:
        user = self.db_session.query(User).filter(User.UserID == user_id).first()
        if not user:
            raise ValueError("User not found")

        if username:
            user.Username = username
        if email:
            user.Email = email
        if role:
            user.Role = role
        if preferences:
            user.Preferences = preferences

        self.db_session.commit()
        return user

    def get_user(self, user_id: int) -> User:
        return self.db_session.query(User).filter(User.UserID == user_id).first()

    def delete_user(self, user_id: int) -> bool:
        # Delete dependent rows in the projects table
        self.db_session.query(Project).filter(Project.OwnerID == user_id).delete()
        
        # Delete the user
        user = self.db_session.query(User).filter(User.UserID == user_id).first()
        if not user:
            raise ValueError("User not found")

        self.db_session.delete(user)
        self.db_session.commit()
        return True