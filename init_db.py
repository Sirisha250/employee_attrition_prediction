import sqlite3

DATABASE = "users.db"

def create_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Create users table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

if __name__ == "__main__":
    create_db()
