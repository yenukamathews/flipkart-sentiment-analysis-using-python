# sentiment_analysis.py

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
from wordcloud import WordCloud

# 2. Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
VISUAL_DIR = os.path.join(BASE_DIR, 'visuals')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# 3. Load Dataset
csv_path = os.path.join(DATA_DIR, 'flipkart_data.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at: {csv_path}")

df = pd.read_csv(csv_path)
print("Dataset loaded with shape:", df.shape)
print("Columns in dataset:", df.columns.tolist())

# 4. Validate required columns
required_columns = ['review', 'rating']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the dataset.")

# 5. Clean and Prepare Data
df.dropna(subset=['review', 'rating'], inplace=True)
df.drop_duplicates(inplace=True)

# Convert ratings to sentiment (4-5 = positive, 1-2 = negative, 3 = neutral to be removed)
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else -1))
df = df[df['sentiment'] != -1]  # Drop neutral

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# 6. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:\n", classification_report(y_test, preds))

# Save best model (Logistic Regression)
best_model = LogisticRegression()
best_model.fit(X_train, y_train)
model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
pickle.dump(best_model, open(model_path, 'wb'))

# 9. Visualizations
# Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment (0=Negative, 1=Positive)")
plt.savefig(os.path.join(VISUAL_DIR, 'sentiment_distribution.png'))

# Word Clouds
pos_text = " ".join(df[df['sentiment'] == 1]['cleaned_review'])
neg_text = " ".join(df[df['sentiment'] == 0]['cleaned_review'])

wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)

wc_pos.to_file(os.path.join(VISUAL_DIR, 'positive_wordcloud.png'))
wc_neg.to_file(os.path.join(VISUAL_DIR, 'negative_wordcloud.png'))

print("\n All tasks completed successfully.")
