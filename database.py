import sqlite3

DB_NAME = "support_pilot.db"


def init_db():
	"""Initializes the SQLite database and creates the tickets table if it doesn't exist."""
	conn = sqlite3.connect(DB_NAME)
	cursor = conn.cursor()

	cursor.execute('''
		CREATE TABLE IF NOT EXISTS tickets (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			description TEXT NOT NULL,
			category TEXT,
			severity TEXT,
			priority TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	''')

	conn.commit()
	conn.close()
	print("Database initialized successfully and 'tickets' table is ready.")


if __name__ == "__main__":
	init_db()
