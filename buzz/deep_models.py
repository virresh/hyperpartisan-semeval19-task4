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

vocab_size = xtrain_count.shape[1]

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

encoded_docs = [one_hot(d, vocab_size) for d in wholeDF['text']]
encoded_docs = pad_sequences(encoded_docs, padding='post')

enctrainX, encvalidX, enctrainY, encvalidY = train_test_split(encoded_docs, wholeDF['label'], test_size=0.2, shuffle=False)

del wholeDF

from sklearn import metrics
def get_accuracy(model, trainX=trainX, testX=validX, trainY=trainY, testY=validY, is_neural_net=False, epochs=5):
    model.fit(trainX, trainY, epochs=epochs)
    predictions = model.predict(testX)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, testY)

from keras import layers, models, optimizers

# Shallow Feed Forward Neural Network
def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Simple CNN
def create_cnn(inp_size):
    # Add an Input Layer
    input_layer = layers.Input((inp_size, ))

    # embedding layer learnt from above
    embedding_layer = layers.Embedding(vocab_size, 200)(input_layer)

    # add dropout on this layer
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

classifier = create_cnn(enctrainX.shape[1])
accuracy_CNN_countvectors = get_accuracy(classifier, trainX=enctrainX, trainY=enctrainY, testX=encvalidX, testY=encvalidY, is_neural_net=True)

classifier = create_model_architecture(xtrain_count.shape[1])
accuracy_NN_countvectors = get_accuracy(classifier, trainX=xtrain_count, testX=xvalid_count, is_neural_net=True)

classifier = create_model_architecture(xtrain_tfidf.shape[1])
accuracy_NN_tfidf = get_accuracy(classifier, trainX=xtrain_tfidf, testX=xvalid_tfidf, is_neural_net=True)

print("CNN, Word Embeddings", accuracy_CNN_countvectors)
print("Simple NN, Count Vectors", accuracy_NN_countvectors)
print("Simple NN, Ngram Level TF IDF Vectors", accuracy_NN_tfidf)
