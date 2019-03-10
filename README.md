# TextClassification-with-MLP

# Text Classification
The objective of this project was to familiarize with text classifiers and especially multilayer perceptron neural network. For this purpose, a [dataset](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools) with customer reviews for restaurants was chosen. 
Reviews are classified in 3 classes (negative, neutral and positive), the dataset includes 2573 reviews which are considered a small number for training. 

## Project Desrciption
Due to the small size of the dataset many candidate hyper-parameters were tuned to get the best results. 
The following classifiers were selected:

1. Intializers
2. Dropout rate
3. Activation function
4. Learning rate reduction
5. Early Stopping
6. Units Dense layer of hidden layers

Additionally two different text representation were tried:
1. TF-IDF Vectors
2. Word Embeddings

### Problems Encoutered
Due to the small size of the dataset it is difficult to draw a conclusion regarding the performance of the various models. 

## Getting Started
# Prerequisites

* This package needs tensorflow installed on GPU.
<br>
other modules:
<br>
* keras
* sklearn
* gensim
* numpy
* pandas
* bs4

### File Description

* There are two different .py files; one for the TF-IDF Vectors and one for the Word Embeddings
* The data files requiered for the train, validation and test datasets can be found [here](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools) , and are under the Restaurant-SB1 category.

### Data Preprocessing

In order to prepare the data for training some preprocessing of the data was necessary. Mainly, to remove punctuation symbols and also use lower case symbols for correct word (string) comparisons. Furthermore, the dataset classes examples were imbalanced. One class (neutral) had very few training examples. The table below shows the particularity of this dataset including validation data after removing na’s.

| Class        | Class Encoding| Instances  |
| -------------|:-------------:| ----------:|
| Negative     | 0 			   | 1696		|
| Neutral      | 1 			   | 104 		|
| Positive     | 2 			   | 1696		|

To overcome the above imbalanced set the following techniques were tested:
* used the validation and train data as one set and used cross validation method which is suggested in such circumstances
* used stratified sampling during the above process to handle the imbalance between the classes
* used a data augmentation technique to ensure that the maximum possible information from the neutral class. The same data were copied another 8 times adding white noise to the tf-idf vectors that were produced.

Furthermore, the following features were added:
a.	One hot encoding of the Categories and Subcategories of the dataset.
b.	The existence or not of capitalized sentences.

## Authors
* **George Vefeidis**
* **Leon Kalderon**
* **Georgia Sarri**
