import pandas as pd
import numpy as np
import re
import string
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. DATA COLLECTION & PREPARATION
print("Loading datasets (Extracting only text column for maximum speed)...")
start_time = time.time()

# Loading only 'text' saves massive RAM and overhead
df_fake = pd.read_csv("Fake.csv", usecols=["text"])
df_true = pd.read_csv("True.csv", usecols=["text"])

# Assign class labels matching the report (Fake = 0, True = 1)
df_fake["label"] = 0
df_true["label"] = 1

# Extract and hold out the last 10 entries for manual verification pool
df_fake_manual_testing = df_fake.tail(10).copy()
df_true_manual_testing = df_true.tail(10).copy()

df_fake.drop(df_fake.tail(10).index, inplace=True)
df_true.drop(df_true.tail(10).index, inplace=True)

# Combine datasets and shuffle
df = pd.concat([df_fake, df_true], axis=0, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna()

# 2. TEXT PREPROCESSING (The 7-Step wordopt Function)
def wordopt(text):
    """
    Executes the exact 7-step text normalization pipeline 
    defined in Section 3.2 of the project report.
    """
    text = text.lower()                                                # Step 1
    text = re.sub(r'\[.*?\]', '', text)                                # Step 2
    text = re.sub(r'\W', ' ', text)                                     # Step 3
    text = re.sub(r'https?://\S+|www\.\S+', '', text)                  # Step 4
    text = re.sub(r'<.*?>+', '', text)                                 # Step 5
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)   # Step 6
    text = re.sub(r'\w*\d\w*', '', text)                               # Step 7
    return text

print("Running 7-step text normalization pipeline...")
df["text"] = df["text"].apply(wordopt)

# 3. FEATURE EXTRACTION & SPLITTING
X = df["text"]
y = df["label"]

# 75/25 Split as required by report Section 3.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Extracting features using TF-IDF (Vocabulary optimized for speed)...")
# max_features=4000 ensures the heavy tree models compile almost instantly
vectorizer = TfidfVectorizer(stop_words='english', max_features=4000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. MULTI-MODEL TRAINING (Logistic, DT, GB, RF)
trained_models = {}
models_config = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42), # Optimized estimators
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)         # Optimized estimators
}

for name, model in models_config.items():
    m_start = time.time()
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model
    
    # Quick internal validation check
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"   -> Done! Accuracy: {acc * 100:.2f}% | Time: {time.time() - m_start:.2f}s")

print(f"\nTotal setup and training finished in {time.time() - start_time:.2f} seconds!")

# 5. USER INTERACTIVE MANUAL TESTING INTERFACE
def manual_testing(news_string):
    # Wrap text in a temporary dataframe structure to apply identical transformations
    test_df = pd.DataFrame({"text": [news_string]})
    test_df["text"] = test_df["text"].apply(wordopt)
    transformed_input = vectorizer.transform(test_df["text"])
    
    print("\n" + "="*55)
    print("             AI MODEL CONSENSUS OUTPUT             ")
    print("="*55)
    
    # Fetch classification output labels from all four algorithms simultaneously
    for name, model in trained_models.items():
        pred = model.predict(transformed_input)
        verdict = "TRUE" if pred[0] == 1 else "FAKE"
        print(f" [{name:22}]: System Output -> {verdict}")
    print("="*55 + "\n")

# Interactive Loop
if __name__ == "__main__":
    print("\nInteractive Console Ready. Enter 'exit' to quit.")
    while True:
        user_input = input("Paste the news text body here:\n> ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Shutting down model interface. System offline.")
            break
        if not user_input.strip():
            continue
        manual_testing(user_input)