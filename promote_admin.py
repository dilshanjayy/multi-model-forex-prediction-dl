import sqlite3
import sys

def promote_to_admin(username: str):
    try:
        conn = sqlite3.connect('trading.db')
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id, username, is_admin FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if not user:
            print(f"Error: User '{username}' not found in database.")
            return
            
        # Update is_admin flag
        cursor.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
        conn.commit()
        
        print(f"Success: User '{username}' has been promoted to Admin.")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python promote_admin.py <username>")
    else:
        promote_to_admin(sys.argv[1])
