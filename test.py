from database import SessionLocal
from user_management import UserManager
from users import User, Base
from project import Project  # Import the Project model
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Initialize the database session
def setup_database():
    db: Session = SessionLocal()
    Base.metadata.create_all(bind=db.bind)  # Ensure the database and tables are created
    return db

def test_create_user():
    db = setup_database()
    user_manager = UserManager(db)
    try:
        # Create a new user
        new_user = user_manager.create_user(
            username="Hanna",
            email="hanna@example.com",
            role="admin",
            preferences="{'theme': 'dark'}"
        )
        print(f"User created: {new_user.UserID}, {new_user.Username}, {new_user.Email}")
    except IntegrityError as e:
        print(f"IntegrityError: {e}")
    finally:
        db.close()

def test_get_user():
    db = setup_database()
    user_manager = UserManager(db)
    try:
        # Retrieve the user with ID 1 (adjust this based on the ID of a user in your DB)
        user = user_manager.get_user(user_id=2)
        if user:
            print(f"User found: {user.UserID}, {user.Username}, {user.Email}")
        else:
            print("User not found.")
    finally:
        db.close()

def test_update_user():
    db = setup_database()
    user_manager = UserManager(db)
    try:
        # Update the user with ID 1 (adjust this based on the ID of a user in your DB)
        updated_user = user_manager.update_user(
            user_id=3,  # Replace with the UserID of the user you want to update
            username="Tesfa",
            email="tesfa@example.com",
            role="user"
        )
        print(f"User updated: {updated_user.UserID}, {updated_user.Username}, {updated_user.Email}")
    except ValueError as e:
        print(f"ValueError: {e}")
    finally:
        db.close()

def test_delete_user():
    db = setup_database()
    user_manager = UserManager(db)
    try:
        # Delete the user with ID 1 (adjust this based on the ID of a user in your DB)
        deleted = user_manager.delete_user(user_id=4)
        if deleted:
            print(f"User deleted: UserID 4")
        else:
            print("User not found or could not be deleted.")
    except ValueError as e:
        print(f"ValueError: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print("Running tests...")
    test_create_user()  # Create a user first
    test_get_user()     # Retrieve the user to verify creation
    test_update_user()  # Update the user's information
    test_get_user()     # Retrieve the user again to verify the update
    test_delete_user()  # Delete the user
    test_get_user()     # Try to retrieve the user again to verify deletion