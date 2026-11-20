import sqlite3

DB_NAME = "support_pilot.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Drop old table to reset schema cleanly
    cursor.execute("DROP TABLE IF EXISTS tickets")
    
    # Create complete tickets schema
    cursor.execute("""
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT NOT NULL,
            category TEXT,
            confidence REAL,
            severity TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database '{DB_NAME}' successfully initialized with full schema!")

if __name__ == "__main__":
    init_db()