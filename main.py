import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('dataset/news.csv')
print(df.shape)
print(df.head())

# grab the labels from the dataframe
labels = df.label
print(labels.head())

# split into train and test 
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print (f'Accuracy: {round(score*100,2)}%')

# Evaluation
disp = ConfusionMatrixDisplay.from_estimator(
    pac, tfidf_test, y_test,
    display_labels=['FAKE', 'REAL'],
    cmap='Blues',
    values_format='d'
)

# ensure a folder exists for saving
os.makedirs("plots", exist_ok=True)

# save plot to the folder
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')

print(classification_report(y_test, y_pred, labels=['FAKE', 'REAL']))

feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
coefs = pac.coef_[0]

top_fake = np.argsort(coefs)[-15:]  # highest positive weights
top_real = np.argsort(coefs)[:15]   # lowest negative weights

plt.figure(figsize=(10,6))
plt.barh(feature_names[top_fake], coefs[top_fake], color='red')
plt.title("Top Words Associated with 'FAKE'")
plt.savefig("plots/FAKE_association.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(10,6))
plt.barh(feature_names[top_real], coefs[top_real], color='green')
plt.title("Top Words Associated with 'REAL'")
plt.savefig("plots/REAL_association.png", dpi=300, bbox_inches='tight')