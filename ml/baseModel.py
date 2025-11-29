# incremental_update_clickbait_model_fixed.py
import re
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# Config / file paths
# -------------------------
OLD_MODEL_PATH = 'fake_news_model_nb_updated.pkl'  # existing NB model
# TF-IDF vectorizer with clickbait
OLD_VECTORIZER_PATH = 'tfidf_vectorizer_naivebayes_clickbait.pkl'
EXTRA_DATA_PATH = r"C:\Users\Tatai\Desktop\python_projects\news_dataset.csv"  # new dataset
IMPROVED_MODEL_PATH = 'fake_news_model_improved.pkl'
IMPROVED_VECTORIZER_PATH = 'tfidf_vectorizer_improved.pkl'

# -------------------------
# 1) Load existing model & vectorizer
# -------------------------
print("Loading existing model and vectorizer...")
model = joblib.load(OLD_MODEL_PATH)
vectorizer = joblib.load(OLD_VECTORIZER_PATH)
print("Loaded model and vectorizer.")

# -------------------------
# 2) Load new dataset
# -------------------------
print("Loading new dataset...")
df_new = pd.read_csv(EXTRA_DATA_PATH)
required_cols = {'text', 'label'}
if not required_cols.issubset(set(df_new.columns)):
    raise ValueError(
        f"{EXTRA_DATA_PATH} must contain columns: {required_cols}")

# Standardize labels
df_new['label'] = df_new['label'].str.lower()

# -------------------------
# 3) Cleaning function
# -------------------------


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)         # keep only word chars and spaces
    text = re.sub(r'\s+', ' ', text)        # collapse multiple spaces
    return text.strip()


df_new['cleaned'] = df_new['text'].apply(clean_text)

# -------------------------
# 4) Add clickbait / sensational feature
# -------------------------
sensational_words = [
    "shocking", "unbelievable", "secret", "exposed", "breaking", "truth", "viral",
    "you won t believe", "government hides", "hidden", "scandal", "conspiracy",
    "alert", "banned", "leaked", "danger", "fake", "hoax", "will replace", "plan revealed",
    "will blow your mind", "you wont believe", "exclusive"
]


def clickbait_score(text: str) -> int:
    return sum(1 for w in sensational_words if w in text.lower())


df_new['clickbait_score'] = df_new['cleaned'].apply(clickbait_score)

df_new['clickbait_score'] = df_new['clickbait_score'] / \
    df_new['clickbait_score'].max()


# -------------------------
# 5) Transform new data using existing TF-IDF
# -------------------------
print("Transforming new data with existing TF-IDF vectorizer...")
X_new_tfidf = vectorizer.transform(df_new['cleaned'])

# Convert clickbait_score to column vector (sparse-compatible)
clickbait_col = np.array(df_new['clickbait_score']).reshape(-1, 1)

# Combine TF-IDF + clickbait
X_new_final = hstack([X_new_tfidf, clickbait_col])
y_new = df_new['label'].values

# -------------------------
# 6) Check feature dimensions
# -------------------------
expected_features = model.feature_log_prob_.shape[1]

if X_new_final.shape[1] != expected_features:
    raise ValueError(
        f"Feature mismatch! Model expects {expected_features}, got {X_new_final.shape[1]}.")

# -------------------------
# 7) Partial_fit update
# -------------------------
print("Updating model with partial_fit ...")
model.partial_fit(X_new_final, y_new, classes=np.array(['fake', 'real']))
print("Model updated (incremental).")

# -------------------------
# 8) Quick evaluation on new data
# -------------------------
y_pred_new = model.predict(X_new_final)
print("\nEvaluation on newly added data:")
print("Accuracy:", accuracy_score(y_new, y_pred_new))
print("\nClassification Report:\n", classification_report(
    y_new, y_pred_new, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_new, y_pred_new))

# -------------------------
# 9) Save improved model & vectorizer copy
# -------------------------
joblib.dump(model, IMPROVED_MODEL_PATH)
joblib.dump(vectorizer, IMPROVED_VECTORIZER_PATH)
print(f"\nImproved model saved to: {IMPROVED_MODEL_PATH}")
print(f"Vectorizer (unchanged) saved to: {IMPROVED_VECTORIZER_PATH}")
