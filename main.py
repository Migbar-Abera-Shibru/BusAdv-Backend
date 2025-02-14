# main.py
from src.database import SessionLocal
from src.managers.user_manager import UserManager

def main():
    # Start a session
    db_session = SessionLocal()
    user_manager = UserManager(db_session)

    # Create a new user
    print("Creating a new user...")
    new_user = user_manager.create_user(username="John Doe", email="john@example.com", role="Admin", preferences="Dark Mode")
    print(f"User created: {new_user.UserID}, {new_user.Username}")

    # Edit the user information
    print("Updating user information...")
    updated_user = user_manager.update_user(new_user.UserID, username="John Smith", email="johnsmith@example.com")
    print(f"User updated: {updated_user.UserID}, {updated_user.Username}, {updated_user.Email}")

    # Fetch and display updated user
    user = user_manager.get_user(updated_user.UserID)
    print(f"User info: ID={user.UserID}, Name={user.Username}, Email={user.Email}, Role={user.Role}")

if __name__ == "__main__":
    main()
