import pandas as pd
import numpy as np
from nltk import word_tokenize
import pickle
import json

wholeDF = pd.DataFrame()
labels = []
ids = []
text = []

with open('data/test_data.txt', 'r') as data_file:
    for line in data_file:
        article_text = line[7:]
        label = int(line[5])
        article_id = line[:4]
        labels.append(label)
        text.append(article_text)
        # text.append(word_tokenize(article_text))
        ids.append(article_id)

wholeDF['label'] = labels
wholeDF['text'] = text
wholeDF['aid'] = ids

del labels, ids, text

# from sklearn.model_selection import train_test_split

# trainX, validX, trainY, validY = train_test_split(wholeDF['text'], wholeDF['label'], test_size=0.2, shuffle=False)

from sklearn.feature_extraction.text import TfidfVectorizer

trainX = wholeDF['text']
validX = wholeDF['label']

# character level uni, bi and tri features
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=5000)
tfidf_vect_ngram_chars.fit(wholeDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(trainX) 
# xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(validX) 

del wholeDF

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(xtrain_tfidf_ngram_chars, validX)
file_name = "library/xbg_model.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(classifier, f)
    pickle.dump(tfidf_vect_ngram_chars, f)
print('Pickle Saved to ', file_name)
