# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:15:19 2019

@author: Georgia Sarri, Leon Kalderon, George Vafeidis
"""
import numpy as np
import pandas as pd
import nltk
import re
import os
from bs4 import BeautifulSoup
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras import backend as K # Importing Keras backend (by default it is Tensorflow)
from keras.layers import Input, Dense # Layers to be used for building our model
from keras.models import Model # The class used to create a model
from keras.optimizers import Adam
from keras.utils import np_utils # Utilities to manipulate numpy arrays
from tensorflow import set_random_seed # Used for reproducible experiments
from tensorflow import keras
from keras.metrics import categorical_accuracy
import tensorflow as tf
import seaborn as sn
import gc
import matplotlib.pyplot as plt

K.set_session(tf.Session())
MODELS_PATH = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\\'
#MODELS_PATH = r'C:\Users\Georgia\Desktop\Assignment3'
model_file_name = os.path.join(MODELS_PATH, 'temp_model.h1')
log_file_name = os.path.join(MODELS_PATH, 'temp.log')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
   
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="orange",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt

def read_data(sentences, flag):
    data = pd.DataFrame(columns=['sentence_id', 'Category', 'Subcategory', 'Target', 'Text', 'Polarity'])
    for sentence in sentences:
        sent_id = sentence.get('id')
        text = sentence.find('text').text
        opinions = sentence.find_all('opinion')
        if(len(opinions)==0):
            data = data.append({'sentence_id': sent_id, 
                                'Category': np.nan, 
                                'Subcategory': np.nan,
                                'Target': np.nan,
                                'Text': text, 
                                'Polarity': np.nan}, ignore_index=True)
        else:
            for opinion in opinions:
                category = opinion.get('category').split('#')[0]
                subcategory = opinion.get('category').split('#')[1]
                polarity = opinion.get('polarity')
                target = 'NA' if flag == 'C' else opinion.get('target')
                data = data.append({'sentence_id': sent_id, 
                                    'Category': category, 
                                    'Subcategory': subcategory, 
                                    'Target': target,
                                    'Text': text, 
                                    'Polarity': polarity}, ignore_index=True)
    return data


def remove_punctuation(words):
    """
    Remove punctuation from list of tokenized words

    :param List[str] words: A list of words(tokens)
    :return List[str] new_words: the list of words with all the punctuations removed.
    """
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def plot_history(hs, epochs, metric):
    plt.clf()
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['font.size'] = 16
    for label in hs:
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric))
        plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric))
    x_ticks = np.arange(0, len(hs) + 1, 5)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.show()

def clean_up(model):
    K.clear_session()
    del model
    gc.collect()  
    
def train_model(train_data,train_labels,
        optimizer,
        epochs=100,
        batch_size=128,
        hidden_layers=0,
        units =400,
        funnel = False,
        hidden_activation='relu',
        output_activation='softmax'):

    # Keras Callbacks
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor = 0.2, patience = 3, min_lr = 1e-6, verbose = 1)
    check_pointer = keras.callbacks.ModelCheckpoint(model_file_name, verbose = 1, save_best_only = True)
    early_stopper = keras.callbacks.EarlyStopping(patience = 8) # Change 4 to 8 in the final run    
    csv_logger = keras.callbacks.CSVLogger(log_file_name)

    np.random.seed(1402) # Define the seed for numpy to have reproducible experiments.
    set_random_seed(1981) # Define the seed for Tensorflow to have reproducible experiments.
    dropout_rate = 0.5
    # Define the input layer.
    input_size = train_data.shape[1]
    input = Input(
        shape=(input_size,),
        name='Input'
    )
    x = input
    # Define the hidden layers.
    for i in range(hidden_layers):
        if funnel:
          layer_units=units // (i+1)
        else: 
          layer_units=units
        x = Dense(
           units=layer_units,
           kernel_initializer='glorot_uniform',
           activation=hidden_activation,
           name='Hidden-{0:d}'.format(i + 1)
        )(x)
        keras.layers.Dropout(x,dropout_rate, seed = 1231)
    # Define the output layer.
    
    output = Dense(
        units=classes,
        kernel_initializer='uniform',
        activation=output_activation,
        name='Output'
    )(x)
    # Define the model and train it.
    model = Model(inputs=input, outputs=output)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=[categorical_accuracy])
    
    keras.backend.get_session().run(tf.global_variables_initializer())
    hs = model.fit(
        x=train_data,
        y=train_labels,
        validation_split=0.1,        
#        validation_split=0.1, # use 10% of the training data as validation data
        epochs=epochs,
        shuffle = True,
        verbose=1,
        batch_size=batch_size,
#        callbacks = [early_stopper, check_pointer,  csv_logger]
        callbacks = [early_stopper, check_pointer, lr_reducer,  csv_logger]
#        callbacks = [early_stopper, nan_terminator, check_pointer, lr_reducer, csv_logger]
        ) 
    print('Finished training.')
    print('------------------')
    model.summary() # Print a description of the model.
    return model, hs


#==== LOAD TRAIN ====
#path = 'C:\\Users\\Georgia.Sarri\\Documents\\Msc\\5th\\TextAnalytics\\Assignmnets\\Untitled Folder\\ABSA16_Laptops_Train_SB1_v2.xml'
path = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\\'
#path = r'C:\Personal\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 3\\'

train_data = pd.DataFrame()

#for file, flag in [('ABSA16_Laptops_Train_SB1_v2.xml', 'C'), ('ABSA16_Restaurants_Train_SB1_v2.xml','R')]:
for file, flag in  [('ABSA16_Restaurants_Train_SB1_v2.xml','R')]:
    data = open(path+file, encoding="utf8").read()
    soup = BeautifulSoup(data, "lxml")
    sentences = soup.find_all("sentence")
    train_data = train_data.append(read_data(sentences, flag))
    
#==== LOAD VALIDATION DATA ====
http = urllib3.PoolManager()
url = 'http://alt.qcri.org/semeval2016/task5/data/uploads/trial-data/english-trial/'
validation_data = pd.DataFrame()

for file, flag in [('restaurants_trial_english_sl.xml', 'R')]:
    response = http.request('GET', url+file)
    soup = BeautifulSoup(response.data, "lxml")
    sentences = soup.find_all("sentence")
    validation_data = validation_data.append(read_data(sentences, flag))
    
train_data = train_data.append(validation_data)
            
train_data.dropna(axis=0, how='any', inplace=True)
X_train = list(itertools.chain.from_iterable(train_data[['Text']].values.tolist()))
Y_train = list(itertools.chain.from_iterable(train_data[['Polarity']].values.tolist()))

train_data[train_data['Polarity']=='neutral']
train_data.groupby(['Polarity']).count()
train_data.columns


X_count = []
for idx,w in enumerate(X_train):
    X_count.extend( [1 if sum([str(c).isupper() for c in w])/len(w) > 0.5 else 0 ])
    w = remove_punctuation(nltk.wordpunct_tokenize(w.lower()))
    X_train[idx] = ' '.join(token for token in w)

X_train_copy = X_train.copy()
    
file1 = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\glove.6B.200d.txt'
file2 = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\gensim_glove_vectors.txt'
#file1 = r'C:\temp\glove.6B\glove.6B.50d.txt'
#file2 = r'C:\temp\glove.6B\gensim_glove_vectors.txt'

glove2word2vec(glove_input_file=file1, word2vec_output_file=file2)
glove_model = KeyedVectors.load_word2vec_format(file2, binary=False)
#wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

### Train Data  ####
train_voc = []
for text in X_train_copy:
    words = text.split()
    for w in words:
        if w not in train_voc:
            train_voc.append(w)
train_voc_set = set(train_voc)
glove_voc_set = set(glove_model.vocab.keys())
print('\n\nTrain set vocabulary subset of glove vocabulary? {}'.format(train_voc_set.issubset(glove_voc_set)))
print('Size of training vocabulary {0:d}'.format(len(train_voc_set)))

text_embedings=list()
#i=0
for idx,text in enumerate(X_train_copy):
    text_in_words =text.split()
    # find only existing words in dictionary
    doc = [word for word in text_in_words if word in glove_model.vocab]
    #Return mean of embeddings
    text_vector = np.mean(glove_model[doc],  axis=0)
    #Create training vector in list form
    text_embedings.append(text_vector)

#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=8000 ,sublinear_tf=True)
#X_train_tfidf = vectorizer.fit_transform(X_train)
vectorizer.fit(X_train_copy)
print(X_train_copy[0])
print(len(vectorizer.vocabulary_))

## First (not most efficient) implementation of centroid taking into account TF-IDF
text_embeddings_tfidf = []
for idx,text in enumerate(X_train_copy):
    text_in_words =text.split()
    
    text_vector = np.zeros(glove_model.vector_size)
    denonimator = 0
    for word in text_in_words:
        if word in glove_model.vocab:#if not found in vocab ignore it
            if word in vectorizer.get_feature_names():
              word_idf = vectorizer.idf_[vectorizer.vocabulary_[word]]
              denonimator += word_idf
              text_vector = text_vector + word_idf * glove_model[word]
    # Calculate Centroids of each text and put it in the list    
    text_embeddings_tfidf.append(text_vector / denonimator)
    

le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)

X_category_dummies = pd.get_dummies(train_data['Category'])
X_subcategory_dummies = pd.get_dummies(train_data['Subcategory'])
X_hasCaps_dummies = pd.get_dummies(X_count)
# X_subcategory_dummies = pd.get_dummies(train_data['Target'])
#X_train_final = sparse.hstack((X_train_tfidf, np.array(X_category_dummies), np.array(X_subcategory_dummies), np.array(X_hasCaps_dummies)))
X_train_final = np.hstack((np.array(text_embeddings_tfidf), np.array(X_category_dummies), np.array(X_subcategory_dummies)))
#X_train_final = X_train_tfidf.copy()

print('Train Data Shape: {}'.format(X_train_final.shape))

### LOAD TEST DATA ###
#path_test = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\EN_REST_SB1_TEST.xml'
path_test = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\\EN_REST_SB1_TEST.xml'

data = open(path_test, encoding="utf8").read()
soup = BeautifulSoup(data, "lxml")
sentences = soup.find_all("sentence")
test_data = read_data(sentences, flag)

test_data.dropna(axis=0, how='any', inplace=True)

X_test = list(itertools.chain.from_iterable(test_data[['Text']].values.tolist()))
Y_test = list(itertools.chain.from_iterable(test_data[['Polarity']].values.tolist()))

X_count_test = []
for idx,w in enumerate(X_test):
    X_count_test.extend( [1 if sum([str(c).isupper() for c in w])/len(w) > 0.5 else 0 ])
    w = remove_punctuation(nltk.wordpunct_tokenize(w.lower()))
    X_test[idx] = ' '.join(token for token in w) 
        
train_voc = []
for text in X_test:
    words = text.split()
    for w in words:
        if w not in train_voc:
            train_voc.append(w)
train_voc_set = set(train_voc)
glove_voc_set = set(glove_model.vocab.keys())
print('\n\nTrain set vocabulary subset of glove vocabulary? {}'.format(train_voc_set.issubset(glove_voc_set)))
print('Size of training vocabulary {0:d}'.format(len(train_voc_set)))

text_embedings_test=list()
for idx,text in enumerate(X_test):
    text_in_words =text.split()
    # find only existing words in dictionary
    doc = [word for word in text_in_words if word in glove_model.vocab]
    #Return mean of embeddings that id tf centroid
    text_vector = np.mean(glove_model[doc],  axis=0)
    #Create training vector in list form
    text_embedings_test.append(text_vector)

#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=8000 ,sublinear_tf=True)
#X_train_tfidf = vectorizer.fit_transform(X_train)
vectorizer.fit(X_test)
print(X_test[0])
print(len(vectorizer.vocabulary_))

## First (not most efficient) implementation of centroid taking into account TF-IDF
text_embeddings_test_tfidf = []
for idx,text in enumerate(X_test):
    text_in_words =text.split()    
    text_vector = np.zeros(glove_model.vector_size)
    denonimator = 0
    for word in text_in_words:
        if word in glove_model.vocab:
            if word in vectorizer.get_feature_names(): #if not found in vocab ignore it
              word_idf = vectorizer.idf_[vectorizer.vocabulary_[word]]
              denonimator += word_idf
              text_vector = text_vector + word_idf * glove_model[word]
    # Calculate Centroids of each text and put it in the list    
    text_embeddings_test_tfidf.append(text_vector / denonimator)

# Choose accordingly which one you would like to train
# before running next section ofr training models
le = preprocessing.LabelEncoder()
le.fit(Y_test)
Y_test = le.transform(Y_test)

X_test_category_dummies = pd.get_dummies(test_data['Category'])
X_test_subcategory_dummies = pd.get_dummies(test_data['Subcategory'])
X_test_hasCaps_dummies = pd.get_dummies(X_count_test)
X_test_final = np.hstack((np.array(text_embeddings_test_tfidf), np.array(X_test_category_dummies), np.array(X_test_subcategory_dummies)))

#X_train_final = text_embedings.copy()
#X_train_final = text_embeddings_tfidf.copy()
#X_test_final = text_embedings_test.copy()
#X_test_final = text_embeddings_test_tfidf.copy()

#X_train_final = X_train_tfidf.copy()

batch_size = 512
classes = 3
epochs = 50

# Using Adam
optimizer = Adam()

# MLP
Y_trainMLP = np_utils.to_categorical(Y_train, classes)
X_train_final_MLP = X_train_final.astype('float32')

mlp_model_adam, mlp_hs_adam = train_model(
    train_data=X_train_final_MLP,
    train_labels=Y_trainMLP,
    optimizer=optimizer,
    epochs=epochs,
    batch_size=batch_size,
    funnel=False,
    hidden_layers=2,
    hidden_activation='relu',
    output_activation='softmax'
)

# Evaluate on test data and show all the results.
Y_test_final = np_utils.to_categorical(Y_test, classes)
X_test_final_MLP = X_test_final
mlp_eval_adam = mlp_model_adam.evaluate(X_test_final_MLP, Y_test_final, verbose=1)
#clean_up(model=mlp_model_adam)


print("Train Loss     : {0:.5f}".format(mlp_hs_adam.history['loss'][-1]))
print("Validation Loss: {0:.5f}".format(mlp_hs_adam.history['val_loss'][-1]))
print("Test Loss      : {0:.5f}".format(mlp_eval_adam[0]))
print("---")
print("Train Accuracy     : {0:.5f}".format(mlp_hs_adam.history['categorical_accuracy'][-1]))
print("Validation Accuracy: {0:.5f}".format(mlp_hs_adam.history['val_categorical_accuracy'][-1]))
print("Test Accuracy      : {0:.5f}".format(mlp_eval_adam[1]))

# mlp_models.append(('Model9',mlp_hs_adam,mlp_eval_adam))

# Plot train and validation error per epoch.
plot_history(hs={'MLP': mlp_hs_adam}, epochs=epochs, metric='loss')
plot_history(hs={'MLP': mlp_hs_adam}, epochs=epochs, metric='categorical_accuracy')


# plot_history(hs={mlp_models[0][0]: mlp_models[0][1]}, epochs=epochs, metric='loss')
# plot_history(hs={mlp_models[0][0]: mlp_models[0][1]}, epochs=epochs, metric='acc')

cm = confusion_matrix(np.argmax(Y_test_final,axis=1),np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1))

df_cm = pd.DataFrame(cm, index = [i for i in np.linspace(0,classes-1,classes,dtype=int)],
                  columns = [i for i in np.linspace(0,classes-1,classes,dtype=int)])
plt.figure(figsize = (4,3))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)

report = classification_report(np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1),
                                           np.argmax(Y_test_final,axis=1),
                                           target_names = le.classes_,digits = 4)
print(report)

