import sqlite3
try:
    conn = sqlite3.connect('trading.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, is_admin FROM users WHERE username = 'test'")
    user = cursor.fetchone()
    if user:
        print(f"User: {user[0]}, is_admin: {user[1]}")
    else:
        print("User 'test' not found.")
finally:
    if conn:
        conn.close()
