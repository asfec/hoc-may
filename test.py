import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_train_data(feature_name, label_name):
    data_feature = pd.read_csv(feature_name, sep=' ', header=None)
    data_feature.columns = ['email', 'word', 'count']
  
    data_label = pd.read_csv(label_name, sep=' ', header=None)
    data_label.columns = ['label']
    num_emails = data_feature['email'].max()
    num_words = data_feature['word'].max()
    X_train = np.zeros((num_emails, num_words))

    for row in data_feature.itertuples():
        X_train[row.email - 1, row.word - 1] = row.count
 
    y_train = data_label['label'].values

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    return X_train, y_train, clf, num_words

def load_test_data(feature_name, label_name, num_words):
    data_feature = pd.read_csv(feature_name, sep=' ', header=None)
    data_feature.columns = ['email', 'word', 'count']

    data_label = pd.read_csv(label_name, sep=' ', header=None)
    data_label.columns = ['label']
    num_emails = data_feature['email'].max()
    max_word_index = data_feature['word'].max()
    num_words = max(num_words, max_word_index)
    X_test = np.zeros((num_emails, num_words))

    for row in data_feature.itertuples():
        X_test[row.email - 1, row.word - 1] = row.count

    y_test = data_label['label'].values

    return X_test, y_test

def run_model(train_feature, train_label, test_feature, test_label):
    X_train, y_train, clf, num_words = load_train_data(train_feature, train_label)
    X_test, y_test = load_test_data(test_feature, test_label, num_words)
    
    if X_test.shape[1] > X_train.shape[1]:
        X_test = X_test[:, :X_train.shape[1]]
    elif X_test.shape[1] < X_train.shape[1]:
        X_test = np.pad(X_test, ((0, 0), (0, X_train.shape[1] - X_test.shape[1])), 'constant')

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Predicted class of test email: {y_pred}')

# data 0 
run_model('train-features.txt', 'train-labels.txt', 'test-features.txt', 'test-labels.txt')

# data 50
run_model('train-features-50.txt', 'train-labels-50.txt', 'test-features.txt', 'test-labels.txt')

# data 100
run_model('train-features-100.txt', 'train-labels-100.txt', 'test-features.txt', 'test-labels.txt')

# data 400
run_model('train-features-400.txt', 'train-labels-400.txt', 'test-features.txt', 'test-labels.txt')