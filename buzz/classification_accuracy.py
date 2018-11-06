import pandas as pd
features = pd.read_csv('prediction.txt')
truth = pd.read_csv('../../BuzzFeedData/overview.csv')
truth['XML'] = truth['XML'].apply(lambda x: int(x[:-4]))
truth = truth[['XML', 'orientation']]
truth['orientation'] = truth['orientation'].apply(lambda x: 0 if x == 'mainstream' else 1)

df = pd.merge(features, truth, left_on='articleId', right_on='XML').drop('XML', axis=1)

X = df.drop(['articleId', 'orientation'], axis=1)
Y = df['orientation']



from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.20)

def get_accuracy(model, trainX=train_X, testX=test_X, trainY=train_y, testY=test_y):
	model.fit(trainX, trainY)
	return model.score(testX, testY)

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
