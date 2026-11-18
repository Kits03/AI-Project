import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ticket_classifier.pkl")

def train_pipeline():
    print("Loading cleaned dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found. Complete Step 2 first.")
        return

    df = pd.read_csv(DATASET_PATH)
    df.dropna(subset=['description', 'category'], inplace=True)

    X = df['description']
    y = df['category']

    # 1. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 2. Text Vectorization using TF-IDF (Unigrams + Bigrams)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    
    print("\nVectorizing text features with TF-IDF...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 3. Model Training: Logistic Regression with Class Weighting
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        C=1.0,
        random_state=42
    )
    
    print("Training Logistic Regression Classifier...")
    model.fit(X_train_tfidf, y_train)

    # 4. Model Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🎯 Overall Model Accuracy: {accuracy * 100:.2f}%")
    print("="*50)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Serialize and Save Model & Vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)
    print(f"\n✅ Model pipeline successfully serialized and saved to '{MODEL_PATH}'")

if __name__ == "__main__":
    train_pipeline()