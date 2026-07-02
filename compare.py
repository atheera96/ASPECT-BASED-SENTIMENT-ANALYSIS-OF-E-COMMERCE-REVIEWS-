import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report # confusion_matrix telah dibuang dari sini

# ========================================================
# 1. Update Clean Data
# ========================================================
print("Loading processed dataset from result.csv...")
if not pd.io.common.file_exists('result.csv'):
    print("Error: Fail 'result.csv' not found.")
    exit()

df = pd.read_csv('result.csv').dropna(subset=['cleaned_review', 'label'])

X = df['cleaned_review'].values.astype('U')
y = df['label'].astype(int)

# Pembahagian Data: 80% Latihan, 20% Ujian (Kekal Konsisten)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================================
# 2. EXTRACTION FEATURES (TF-IDF)
# ========================================================
print(" Extracting features using TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========================================================
# 3. Model Comparison
# ========================================================
models = {
    'Naïve Bayes': MultinomialNB(),
    'Support Vector Machine (SVM)': LinearSVC(class_weight='balanced', random_state=42, max_iter=2000),
    'Logistic Regression (LR)': LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42)
}

# Result
results_summary = []

print("\n Starting Model Training and Evaluation...\n")

# Loop for train and test model 
for name, model in models.items():
    print(f"=================== {name} ===================")
    # train model
    model.fit(X_train_tfidf, y_train)
    
    # predict test
    y_pred = model.predict(X_test_tfidf)
    
    # Kira metrik
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy Score: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['0 (Positive)', '1 (Neutral)', '2 (Negative)']))
    print("-" * 50)
    
    # Result
    results_summary.append({
        'Model': name,
        'Accuracy (%)': round(acc * 100, 2)
    })

# ========================================================
# 4. Comparison Result
# ========================================================
print("\n SUMMARY ")
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))