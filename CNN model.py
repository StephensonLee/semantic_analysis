import pandas as pd
import numpy as np
import tensorflow as tf


# Importing Reviews dataset
reviews = pd.read_csv("review_Kindle_Test.csv")
print(reviews.shape)
reviews = reviews.dropna()
print(reviews.shape)
print(reviews.head(5))
print(reviews.isnull().values.any())

# Data preprocessing
import re
from nltk.corpus import stopwords
def preprocess(sentence):
    sentence = sentence.lower()

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    stopwords_list = set(stopwords.words('english'))
    pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Data preprocessing, clean up data, remove useless part.
import matplotlib.pyplot as plt

# undersampling
y = reviews['overall']
nums=y.value_counts().sort_index().values
samples = [20000,20000,20000,20000,20000]
revs = []
ovls = []
X = reviews['reviewText']
for i in range(len(y.values)):
    if y.values[i]==1 and samples[0]>0:
        revs.append(X.values[i])
        ovls.append(0)
        samples[0]-=1
    elif y.values[i]==2 and samples[1]>0:
        revs.append(X.values[i])
        ovls.append(1)
        samples[1] -= 1
    elif y.values[i]==3 and samples[2]>0:
        revs.append(X.values[i])
        ovls.append(2)
        samples[2] -= 1
    elif y.values[i]==4 and samples[3]>0:
        revs.append(X.values[i])
        ovls.append(3)
        samples[3] -= 1
    elif y.values[i]==5 and samples[4]>0:
        revs.append(X.values[i])
        ovls.append(4)
        samples[4] -= 1

y = np.array(ovls)
X = []
sentences = list(revs)
for sen in sentences:
    X.append(preprocess(sen))
print('X:',len(X),'y:',len(y))

# Split dataset into train dataset and test dataset
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[:10])

# Preparing embedding layer, converts sentences to their numeric form
from keras.preprocessing.text import Tokenizer
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)
print(X_train[10])

# Adding 1 to store dimensions for words for which no pretrained word embeddings exist
vocab_length = len(word_tokenizer.word_index) + 1
print('vocab_length: ',vocab_length)

# Padding all reviews to fixed length 100
from keras.utils import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Import 100 dimension Load GloVe word embeddings and create an Embeddings Dictionary
embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

# Create Embedding Matrix having 100 columns
embedding_matrix = np.zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print('embedding_matrix.shape:')
print(embedding_matrix.shape)


import pickle
with open('embedding_matrix.pkl', 'wb') as fp:
    pickle.dump(embedding_matrix, fp)
    print('embedding_matrix saved successfully to file')
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
with open('embedding_matrix.pkl', 'rb') as fp:
    embedding_matrix = pickle.load(fp)


# from keras.models import Sequential,Model
# from keras.layers import Activation, Dropout ,Dense, LSTM, Embedding,Dropout,SpatialDropout1D,MaxPooling1D,GRU,BatchNormalization
# from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
# from keras.layers import Input,Bidirectional,GlobalAveragePooling1D,concatenate,LeakyReLU
# from keras import regularizers
# from keras import backend as K

# Neural Network architecture
# Model constants.
max_features = len(word_tokenizer.word_index) + 1
embedding_dims = 100
maxlen = 100
filters = 100
kernel_size = 3
hidden_dims = 100


# CNN with max pooling imeplementation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
print('Build model...')
model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print(model.summary())

# Model training
cnn_model_history = model.fit(X_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(X_test, y_test))

# Predictions on the Test Set
score = model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
y_pred = model.predict(X_test)


# confusion_matrix = tf.math.confusion_matrix(y_test, y_pred)
# print(confusion_matrix)

# Model Performance Charts

plt.plot(cnn_model_history.history['acc'])
plt.plot(cnn_model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(cnn_model_history.history['loss'])
plt.plot(cnn_model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()