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

    return X_train, y_train  , clf

def load_test_data(feature_name, label_name,):
    data_feature = pd.read_csv(feature_name, sep=' ', header=None)
    data_feature.columns = ['email', 'word', 'count']

    data_label = pd.read_csv(label_name, sep=' ', header=None)
    data_label.columns = ['label']

    num_test_emails = data_feature['email'].max()
    num_words = data_feature['word'].max()

    X_test = np.zeros((num_test_emails, num_words))

    for row in data_feature.itertuples():
        X_test[row.email - 1, row.word - 1] = row.count
    y_test = data_label['label'].values

    return X_test, y_test

def evaluate_model(train_feature, train_label, test_feature, test_label):
    X_train, y_train, clf = load_train_data(train_feature, train_label)
    X_test, y_test = load_test_data(test_feature, test_label)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Predicted class of test email: {y_pred}')
#    data 0 
evaluate_model('train-features.txt', 'train-labels.txt', 'test-features.txt', 'test-labels.txt')

# data 50
evaluate_model('train-features-50.txt', 'train-labels-50.txt', 'test-features.txt', 'test-labels.txt')

# data 100
evaluate_model('train-features-100.txt', 'train-labels-100.txt', 'test-features.txt', 'test-labels.txt')

# data 400
evaluate_model('train-features-400.txt', 'train-labels-400.txt', 'test-features.txt', 'test-labels.txt')
