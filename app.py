from flask import Flask, render_template, request
import joblib
import sqlite3
import os

app = Flask(__name__)

DB_NAME = "support_pilot.db"
MODEL_PATH = os.path.join("models", "ticket_classifier.pkl")

# Initialize global variables
model = None
vectorizer = None

def init_db():
    """Automatically ensures the database and table exist on startup."""
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

# Run database initialization immediately when app starts
init_db()

# Load the dictionary bundle saved by train_model.py
if os.path.exists(MODEL_PATH):
    print(f"Loading model bundle from {MODEL_PATH}...")
    artifact = joblib.load(MODEL_PATH)
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    print("SUCCESS: Model and Vectorizer loaded successfully!")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}! Please run train_model.py first.")

def get_recent_tickets():
    """Fetches the last 5 submitted tickets from SQLite"""
    if not os.path.exists(DB_NAME):
        return []
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM tickets ORDER BY id DESC LIMIT 5')
    tickets = cursor.fetchall()
    conn.close()
    return tickets

def get_ticket_stats():
    """Returns total ticket count"""
    if not os.path.exists(DB_NAME):
        return 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM tickets')
    count = cursor.fetchone()[0]
    conn.close()
    return count

def save_ticket_to_db(description, category, severity, priority):
    """Helper function to log submitted tickets into SQLite"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO tickets (description, category, severity, priority)
        VALUES (?, ?, ?, ?)
    ''', (description, category, severity, priority))
    conn.commit()
    ticket_id = cursor.lastrowid
    conn.close()
    return ticket_id

@app.route('/')
def home():
    recent_tickets = get_recent_tickets()
    total_tickets = get_ticket_stats()
    return render_template('index.html', recent_tickets=recent_tickets, total_tickets=total_tickets)

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer
    if request.method == 'POST':
        # 1. Capture both Title and Description from the form
        ticket_title = request.form.get('ticket_title', '')
        ticket_desc = request.form.get('ticket', '')
        severity = request.form.get('severity', 'Medium')
        priority = request.form.get('priority', 'Normal')
        
        # Combine title and description for richer AI classification text
        full_ticket_text = f"{ticket_title}: {ticket_desc}"
        
        if not ticket_desc.strip():
            return render_template('index.html', 
                                   prediction="Error: Ticket description cannot be empty!", 
                                   ticket_text="", 
                                   ticket_id=0,
                                   recent_tickets=get_recent_tickets(),
                                   total_tickets=get_ticket_stats())

        if model is None or vectorizer is None:
            return render_template('index.html', 
                                   prediction="Error: Model bundle not loaded correctly! Run train_model.py first.", 
                                   ticket_text=full_ticket_text, 
                                   ticket_id=0,
                                   recent_tickets=get_recent_tickets(),
                                   total_tickets=get_ticket_stats())

        # 2. Vectorize and Predict using combined text
        ticket_vector = vectorizer.transform([full_ticket_text])
        predicted_category = model.predict(ticket_vector)[0]
        
        # 3. Store the combined text in SQLite
        ticket_id = save_ticket_to_db(full_ticket_text, predicted_category, severity, priority)
        
        # 4. Refresh Dashboard Data
        recent_tickets = get_recent_tickets()
        total_tickets = get_ticket_stats()
        
        return render_template('index.html', 
                               prediction=predicted_category, 
                               ticket_text=full_ticket_text, 
                               ticket_id=ticket_id,
                               recent_tickets=recent_tickets,
                               total_tickets=total_tickets)

if __name__ == '__main__':
    app.run(debug=True, port=5000)