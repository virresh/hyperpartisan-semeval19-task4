import pandas as pd
features = pd.read_csv('prediction.txt')
truth = pd.read_csv('../../../BuzzFeedData/overview.csv')
truth['XML'] = truth['XML'].apply(lambda x: int(x[:-4]))
truth = truth[['XML', 'orientation']]
truth['orientation'] = truth['orientation'].apply(lambda x: 0 if x == 'mainstream' else 1)

df = pd.merge(features, truth, left_on='articleId', right_on='XML').drop('XML', axis=1)

X = df.drop(['articleId', 'orientation'], axis=1)
Y = df['orientation']

"""
These models use features such as sentiment of article etc etc
"""

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import json

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.20)

# def get_accuracy(model, trainX=train_X, testX=test_X, trainY=train_y, testY=test_y):
# 	model.fit(trainX, trainY)
# 	return model.score(testX, testY)
def get_accuracy(model, trainX=train_X, testX=test_X, trainY=train_y, testY=test_y):
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    accuracy = metrics.accuracy_score(predictions, testY)
    precision_micro = metrics.precision_score(testY, predictions, average='micro')
    precision_binary = metrics.precision_score(testY, predictions)
    precision_macro = metrics.precision_score(testY, predictions, average='macro') 
    recall_micro =  metrics.recall_score(testY, predictions, average='micro')
    recall_binary =  metrics.recall_score(testY, predictions)
    recall_macro =  metrics.recall_score(testY, predictions, average='macro')
    f1_micro = metrics.f1_score(testY, predictions, average='micro')
    f1_binary = metrics.f1_score(testY, predictions)
    f1_macro = metrics.f1_score(testY, predictions, average='macro')
    d = {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'precision_binary': precision_binary,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_binary': recall_binary,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_binary': f1_binary,
        'f1_macro': f1_macro
    }
    return json.dumps(d, indent=4)

def cross_validate_kfold(model, X=X, Y=Y, k=3):
	return cross_val_score(model, X, Y, cv=k)


# KNN
print('KNN, k=50, Accuracy:', get_accuracy(KNeighborsClassifier(n_neighbors=50)))
temp_arr = cross_validate_kfold(KNeighborsClassifier(n_neighbors=50))
print('KNN, k=50, 3-Cross Fold Accuracy:', temp_arr, 'Accuracy Mean', np.mean(temp_arr))

# NaiveBayes
print('Gaussian Naive Bayes, k=50, Accuracy:', get_accuracy(GaussianNB()))
temp_arr = cross_validate_kfold(GaussianNB())
print('Gaussian Naive Bayes, k=50, 3-Cross Fold Accuracy:', temp_arr, 'Accuracy Mean', np.mean(temp_arr))

# Random Forest
print('Random Forest, k=50, Accuracy:', get_accuracy(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)))
temp_arr = cross_validate_kfold(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0))
print('Random Forest, k=50, 3-Cross Fold Accuracy:', temp_arr, 'Accuracy Mean', np.mean(temp_arr))

# LogisticRegression
print('Logistic Regression, k=50, Accuracy:', get_accuracy(LogisticRegression()))
temp_arr = cross_validate_kfold(LogisticRegression())
print('Logistic Regression, k=50, 3-Cross Fold Accuracy:', temp_arr, 'Accuracy Mean', np.mean(temp_arr))

# SVM
print('SVM (Linear SVC), k=50, Accuracy:', get_accuracy(LinearSVC(tol=1e-1)))
temp_arr = cross_validate_kfold(LinearSVC(tol=1e-1))
print('SVM (Linear SVC), k=50, 3-Cross Fold Accuracy:', temp_arr, 'Accuracy Mean', np.mean(temp_arr))
