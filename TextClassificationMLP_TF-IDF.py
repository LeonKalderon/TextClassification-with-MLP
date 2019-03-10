# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:15:19 2019

@author: Georgia Sarri, Leon Kalderon, George Vafeidis
"""

import numpy as np
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn import preprocessing
from scipy import sparse
from keras import backend as K # Importing Keras backend (by default it is Tensorflow)
from keras.layers import Input, Dense # Layers to be used for building our model
from keras.models import Model # The class used to create a model
from keras.optimizers import Adam
from keras.utils import np_utils # Utilities to manipulate numpy arrays
from tensorflow import set_random_seed # Used for reproducible experiments
from tensorflow import keras
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

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
    x_ticks = np.arange(0, epochs + 1, epochs / 10)
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
        units = 300,
        funnel = False,
        hidden_activation='relu',
        output_activation='softmax'):
    
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
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hs = model.fit(
        x=train_data,
        y=train_labels,
        validation_split=0.1, # use 10% of the training data as validation data
        epochs=epochs,
        verbose=1,
        batch_size=batch_size
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
    
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)

idx = np.where(np.array(Y_train) == 'neutral')

x_train_neutral = X_train_tfidf[idx]
X_category_neutral = []#list(train_data['Category'].iloc[idx])
X_subcategory_neutral = []#list(train_data['Subcategory'].iloc[idx])
for i in range(8):
    X_train_tmp = x_train_neutral.copy()
    for index in range(len(X_train_tmp.data)):
        tmp = np.random.normal(0.0,0.05)
        X_train_tmp.data[index] += tmp if tmp > 0 else 0
    X_train_tfidf = sparse.vstack([X_train_tfidf, X_train_tmp])
    #X_train_tfidf = sparse.vstack([X_train_tfidf, x_train_neutral])
    X_category_neutral.extend(train_data['Category'].iloc[idx])
    X_subcategory_neutral.extend(train_data['Subcategory'].iloc[idx])
    
Y_train.extend(['neutral']*8*(sum(train_data['Polarity']=='neutral')))

le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)

X_category_dummies = pd.get_dummies(np.append(train_data['Category'], np.array(X_category_neutral)))
X_subcategory_dummies = pd.get_dummies(np.append(train_data['Subcategory'], np.array(X_subcategory_neutral)))
X_hasCaps_dummies = pd.get_dummies(X_count)
# X_subcategory_dummies = pd.get_dummies(train_data['Target'])
#X_train_final = sparse.hstack((X_train_tfidf, np.array(X_category_dummies), np.array(X_subcategory_dummies), np.array(X_hasCaps_dummies)))
X_train_final = sparse.hstack((X_train_tfidf, np.array(X_category_dummies), np.array(X_subcategory_dummies)))
#X_train_final = X_train_tfidf.copy()

print('Train Data Shape: {}'.format(X_train_final.shape))

### LOAD TEST DATA ###
#path_test = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\EN_REST_SB1_TEST.xml'
path_test = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\EN_REST_SB1_TEST.xml'

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
        
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
X_test_tfidf = vectorizer.fit_transform(X_test)


le = preprocessing.LabelEncoder()
le.fit(Y_test)
Y_test = le.transform(Y_test)

X_test_category_dummies = pd.get_dummies(test_data['Category'])
X_test_subcategory_dummies = pd.get_dummies(test_data['Subcategory'])
X_test_hasCaps_dummies = pd.get_dummies(X_count_test)
X_test_final = sparse.hstack((X_test_tfidf, np.array(X_test_category_dummies), np.array(X_test_subcategory_dummies)))

#X_train_final = X_train_tfidf.copy()

batch_size = 128
classes = 3
epochs = 50

# Using Adam
optimizer = Adam()

# MLP
Y_trainMLP = np_utils.to_categorical(Y_train, classes)
X_train_final_MLP = X_train_final.astype('float32').tocsr()

mlp_model_adam, mlp_hs_adam = train_model(
    train_data=X_train_final_MLP,
    train_labels=Y_trainMLP,
    optimizer=optimizer,
    epochs=epochs,
    batch_size=batch_size,
    funnel=True,
    hidden_layers=2,
    hidden_activation='relu',
    output_activation='softmax'
)

# Evaluate on test data and show all the results.
Y_test_final = np_utils.to_categorical(Y_test, classes)
X_test_final_MLP = X_test_final.tocsr()
mlp_eval_adam = mlp_model_adam.evaluate(X_test_final_MLP, Y_test_final, verbose=1)
#clean_up(model=mlp_model_adam)


print("Train Loss     : {0:.5f}".format(mlp_hs_adam.history['loss'][-1]))
print("Validation Loss: {0:.5f}".format(mlp_hs_adam.history['val_loss'][-1]))
print("Test Loss      : {0:.5f}".format(mlp_eval_adam[0]))
print("---")
print("Train Accuracy     : {0:.5f}".format(mlp_hs_adam.history['acc'][-1]))
print("Validation Accuracy: {0:.5f}".format(mlp_hs_adam.history['val_acc'][-1]))
print("Test Accuracy      : {0:.5f}".format(mlp_eval_adam[1]))

# mlp_models.append(('Model9',mlp_hs_adam,mlp_eval_adam))

# Plot train and validation error per epoch.
plot_history(hs={'MLP': mlp_hs_adam}, epochs=epochs, metric='loss')
plot_history(hs={'MLP': mlp_hs_adam}, epochs=epochs, metric='acc')


# plot_history(hs={mlp_models[0][0]: mlp_models[0][1]}, epochs=epochs, metric='loss')
# plot_history(hs={mlp_models[0][0]: mlp_models[0][1]}, epochs=epochs, metric='acc')

confusion_matrix1 = confusion_matrix(np.argmax(Y_test_final,axis=1),np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1))

