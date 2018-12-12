import pandas as pd
import numpy as np
from nltk import word_tokenize
import pickle
import json

classifier = None
feature_extractor = None

with open('xbg_model.pkl', 'rb') as f:
	classifier = pickle.load(f)
	feature_extractor = pickle.load(f)

def predict_func(article):
	feats = feature_extractor.transform(article_text)
	predictions = classifier.predict(testX)
	return predictions[0]
