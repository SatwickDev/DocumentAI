import os
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from training_dataset import datasets  # Import dataset from the module

# ---------------- Model Directory ----------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# ---------------- Load Training Data ----------------
texts, labels = [], []
for category, texts_list in datasets.items():
    texts.extend(texts_list)
    labels.extend([category] * len(texts_list))

print(f"Training data loaded: {len(texts)} samples across {len(set(labels))} categories")
print("Category distribution:")
for category in set(labels):
    count = labels.count(category)
    print(f"  - {category}: {count} samples")

# ---------------- IMPROVED: Enhanced TF-IDF Settings ----------------
# More aggressive feature extraction for document classification
vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased from 10000
    ngram_range=(1, 4),  # Increased from (1,3) to capture more patterns
    min_df=1,           # Keep rare terms that might be document-specific
    max_df=0.95,        # Remove very common terms
    lowercase=True,
    stop_words=None,    # Keep all words including common ones for business docs
    sublinear_tf=True,  # Apply sublinear scaling
    strip_accents='unicode'
)

print("Vectorizing training data...")
X = vectorizer.fit_transform(texts)
print(f"Feature matrix shape: {X.shape}")

# ---------------- IMPROVED: Better Model Configuration ----------------
# Enhanced LogisticRegression with better parameters
model = LogisticRegression(
    max_iter=5000,      # Increased iterations
    C=1.0,              # Reduced from 2.0 for better generalization
    class_weight='balanced',  # Handle class imbalance
    random_state=42,    # For reproducibility
    solver='lbfgs'  # Better for small datasets
)

print("Training model...")
model.fit(X, labels)

# ---------------- NEW: Model Validation ----------------
# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# Train on split data
model_val = LogisticRegression(
    max_iter=5000,
    C=1.0,
    class_weight='balanced',
    random_state=42,
    solver='liblinear'
)
model_val.fit(X_train, y_train)

# Evaluate
y_pred = model_val.predict(X_test)
print("\n" + "="*50)
print("MODEL VALIDATION RESULTS:")
print("="*50)
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, labels, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ---------------- NEW: Test with Sample Texts ----------------
print("\n" + "="*50)
print("TESTING ON SAMPLE TEXTS:")
print("="*50)

test_samples = [
    "PURCHASE ORDER PO-2025-001 BUYER: XYZ Ltd. SUPPLIER: ABC Electronics. ITEM: Laptops. QUANTITY: 50 UNITS.",
    "APPLICATION FOR LETTER OF CREDIT NO: LC-2025-001. APPLICANT: XYZ Ltd. BENEFICIARY: ABC EXPORTS.",
    "BANK GUARANTEE BG-001 ISSUED BY ABC BANK. AMOUNT: USD 100,000. BENEFICIARY: MINISTRY OF WORKS.",
    "PROFORMA INVOICE PI-2025-044 ISSUED TO ABC Importers LLC. SHIPMENT TERMS: CIF Jebel Ali."
]

test_expected = ["Purchase Order", "LC Application Form", "Bank Guarantee", "Proforma Invoice"]

for i, (sample, expected) in enumerate(zip(test_samples, test_expected)):
    X_sample = vectorizer.transform([sample])
    pred = model.predict(X_sample)[0]
    probs = model.predict_proba(X_sample)[0]
    max_prob = probs.max()

    print(f"\nSample {i+1}:")
    print(f"Text: {sample[:60]}...")
    print(f"Expected: {expected}")
    print(f"Predicted: {pred}")
    print(f"Confidence: {max_prob:.3f}")
    print(f"Match: {'YES' if pred == expected else 'NO'}")

# ---------------- Save Model ----------------
# Save model files to model directory with correct names
joblib.dump(model, os.path.join(model_dir, "classifier.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
print("Model saved successfully!")

# ---------------- NEW: Save Model Metadata ----------------
model_info = {
    "training_samples": len(texts),
    "categories": list(set(labels)),
    "feature_count": X.shape[1],
    "cv_accuracy": cv_scores.mean(),
    "vectorizer_params": vectorizer.get_params(),
    "model_params": model.get_params()
}


with open(os.path.join(model_dir, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2, default=str)

print("\nModel training completed successfully!")
print(f"Training accuracy: {model.score(X, labels):.3f}")
print(f"Cross-validation accuracy: {cv_scores.mean():.3f}")
print(f"Feature count: {X.shape[1]}")
print("\nModel files saved successfully.")
print(f"  - {model_dir}/model_info.json")
