import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load training features
train_features = pd.read_csv('train-features.txt', sep=' ', header=None)
train_features.columns = ['email_index', 'word_index', 'count']

# Load training labels
train_labels = pd.read_csv('train-labels.txt', sep=' ', header=None)
train_labels.columns = ['label']

# Convert features to a matrix
num_emails = train_features['email_index'].max()
num_words = train_features['word_index'].max()
X_train = np.zeros((num_emails, num_words))

for row in train_features.itertuples():
    X_train[row.email_index - 1, row.word_index - 1] = row.count

y_train = train_labels['label'].values

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Load test features
test_features = pd.read_csv('test-features.txt', sep=' ', header=None)
test_features.columns = ['email_index', 'word_index', 'count']

# Load test labels
test_labels = pd.read_csv('test-labels.txt', sep=' ', header=None)
test_labels.columns = ['label']

# Convert test features to a matrix
num_test_emails = test_features['email_index'].max()
X_test = np.zeros((num_test_emails, num_words))
for row in test_features.itertuples():
    X_test[row.email_index - 1, row.word_index - 1] = row.count

y_test = test_labels['label'].values

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Predicted class of  test email: {y_pred}')
print(f'Probability of test email in each class: {clf.predict_proba(X_test)}')