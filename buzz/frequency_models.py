import pandas as pd
import numpy as np
from nltk import word_tokenize

wholeDF = pd.DataFrame()
labels = []
ids = []
text = []

with open('test_data.txt', 'r') as data_file:
    for line in data_file:
        article_text = line[7:]
        label = line[5]
        article_id = line[:4]
        labels.append(label)
        text.append(article_text)
        # text.append(word_tokenize(article_text))
        ids.append(article_id)

wholeDF['label'] = labels
wholeDF['text'] = text
wholeDF['aid'] = ids

del labels, ids, text

from sklearn.model_selection import train_test_split

trainX, validX, trainY, validY = train_test_split(wholeDF['text'], wholeDF['label'], test_size=0.2, shuffle=False)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# count vector representation
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(wholeDF['text'])
xtrain_count =  count_vect.transform(trainX)
xvalid_count =  count_vect.transform(validX)

# unigram, bigram and trigram features
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, ngram_range=(1,3))
tfidf_vect_ngram.fit(wholeDF['text'])
xtrain_tfidf =  tfidf_vect_ngram.transform(trainX)
xvalid_tfidf =  tfidf_vect_ngram.transform(validX)

# character level uni, bi and tri features
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=5000)
tfidf_vect_ngram_chars.fit(wholeDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(trainX) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(validX) 

del wholeDF

from sklearn import metrics
def get_accuracy(model, trainX=trainX, testX=validX, trainY=trainY, testY=validY, is_neural_net=False):
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, testY)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

print('Count Vectors:')
print('KNN, Count Vector Accuracy:', get_accuracy(KNeighborsClassifier(n_neighbors=50), trainX=xtrain_count, testX=xvalid_count))
print('Gaussian Naive Bayes, Count Vector Accuracy:', get_accuracy(GaussianNB(), trainX=xtrain_count.todense(), testX=xvalid_count.todense()))
print('Random Forest, Count Vector Accuracy:', get_accuracy(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), trainX=xtrain_count, testX=xvalid_count))
print('Logistic Regression, Count Vector Accuracy:', get_accuracy(LogisticRegression(solver='lbfgs'), trainX=xtrain_count, testX=xvalid_count))
print('SVM (Linear SVC), Count Vector Accuracy:', get_accuracy(LinearSVC(tol=1e-1), trainX=xtrain_count, testX=xvalid_count))
print('xgboost, Count Vector Accuracy:', get_accuracy(XGBClassifier(), trainX=xtrain_count.tocsc(), testX=xvalid_count.tocsc()))
print('\n')

print('Word Vectors:')
print('KNN, Word Vectors Accuracy:', get_accuracy(KNeighborsClassifier(n_neighbors=50), trainX=xtrain_tfidf, testX=xvalid_tfidf))
print('Gaussian Naive Bayes, Word Vectors Accuracy:', get_accuracy(GaussianNB(), trainX=xtrain_tfidf.todense(), testX=xvalid_tfidf.todense()))
print('Random Forest, Word Vectors Accuracy:', get_accuracy(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), trainX=xtrain_tfidf, testX=xvalid_tfidf))
print('Logistic Regression, Word Vectors Accuracy:', get_accuracy(LogisticRegression(solver='lbfgs'), trainX=xtrain_tfidf, testX=xvalid_tfidf))
print('SVM (Linear SVC), Word Vectors Accuracy:', get_accuracy(LinearSVC(tol=1e-1), trainX=xtrain_tfidf, testX=xvalid_tfidf))
print('xgboost, Word Vectors Accuracy:', get_accuracy(XGBClassifier(), trainX=xtrain_tfidf, testX=xvalid_tfidf))
print('\n')

print('Character Vectors:')
print('KNN, Character Vectors Accuracy:', get_accuracy(KNeighborsClassifier(n_neighbors=50), trainX=xtrain_tfidf_ngram_chars, testX=xvalid_tfidf_ngram_chars))
print('Gaussian Naive Bayes, Character Vectors Accuracy:', get_accuracy(GaussianNB(), trainX=xtrain_tfidf_ngram_chars.todense(), testX=xvalid_tfidf_ngram_chars.todense()))
print('Random Forest, Character Vectors Accuracy:', get_accuracy(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), trainX=xtrain_tfidf_ngram_chars, testX=xvalid_tfidf_ngram_chars))
print('Logistic Regression, Character Vectors Accuracy:', get_accuracy(LogisticRegression(solver='lbfgs'), trainX=xtrain_tfidf_ngram_chars, testX=xvalid_tfidf_ngram_chars))
print('SVM (Linear SVC), Character Vectors Accuracy:', get_accuracy(LinearSVC(tol=1e-1), trainX=xtrain_tfidf_ngram_chars, testX=xvalid_tfidf_ngram_chars))
print('xgboost, Character Vectors Accuracy:', get_accuracy(XGBClassifier(), trainX=xtrain_tfidf_ngram_chars, testX=xvalid_tfidf_ngram_chars))
print('\n')
