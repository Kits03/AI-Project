import os
import joblib

MODEL_PATH = os.path.join("models", "ticket_classifier.pkl")

# Operational mapping for Severity and SLA Priority based on predicted category
SEVERITY_MAP = {
    "System": ("Critical", "P1"),
    "VPN": ("High", "P2"),
    "Network": ("High", "P2"),
    "Password": ("Medium", "P3"),
    "Software": ("Medium", "P3"),
    "Hardware": ("Low", "P4")
}

def load_classifier():
    """Loads the serialized TF-IDF vectorizer and Logistic Regression model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run train_model.py first.")
    
    artifact = joblib.load(MODEL_PATH)
    return artifact["vectorizer"], artifact["model"]

def predict_ticket(description, title=""):
    """
    Accepts raw ticket title and description, vectorizes text with TF-IDF,
    predicts the category, calculates prediction confidence, and maps Severity/Priority.
    """
    vectorizer, model = load_classifier()
    
    # Combine title and description into a single text representation
    full_text = f"{title}. {description}".strip() if title else description
    
    # Transform text using fitted TF-IDF Vectorizer
    text_tfidf = vectorizer.transform([full_text])
    
    # Extract prediction probability distribution across classes
    probabilities = model.predict_proba(text_tfidf)[0]
    classes = model.classes_
    
    max_idx = probabilities.argmax()
    predicted_category = str(classes[max_idx])
    confidence_score = float(probabilities[max_idx])
    
    # High-impact rule override for emergency keywords (BSOD, server down, outage)
    lower_text = full_text.lower()
    if any(term in lower_text for term in ["bsod", "blue screen", "outage", "system down", "server down"]):
        severity, priority = "Critical", "P1"
    else:
        severity, priority = SEVERITY_MAP.get(predicted_category, ("Medium", "P3"))
        
    return {
        "category": predicted_category,
        "confidence": round(confidence_score, 4),
        "severity": severity,
        "priority": priority
    }

if __name__ == "__main__":
    # Test run with a sample issue description
    sample_description = "Computer displaying Blue Screen of Death (BSOD) stop code memory management."
    test_result = predict_ticket(sample_description)
    print("--- Test Inference Output ---")
    print(test_result)