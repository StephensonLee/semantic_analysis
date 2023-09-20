import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop

# Importing Reviews dataset
reviews = pd.read_csv("review_Kindle_Test.csv")
print(reviews.shape)
reviews = reviews.dropna()
print(reviews.shape)
print(reviews.head(5))
print(reviews.isnull().values.any())


import re
from nltk.corpus import stopwords
# Data preprocessing
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
X = []
sentences = list(reviews['reviewText'])
for sen in sentences:
    X.append(preprocess(sen))

y = reviews['overall']
ax = y.value_counts().sort_index().plot(kind = 'bar',title = 'Count of Review Score', figsize=(10,5))
ax.set_xlabel('Review Scores')
plt.show()
y = np.array(list(y))
print(y)

counter = {}
for i in y:
    if i in counter:
        counter[i]+=1
    else:
        counter[i]=1
print(counter)

# Split dataset into train dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# y_train_cat = to_categorical(y_train)
# y_test_cat = to_categorical(y_test)
# print(y_train_cat[:3])

# Preparing embedding layer, converts sentences to their numeric form
from keras.preprocessing.text import Tokenizer
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)
print(X_train[10])

# import io
# import json
#
# # Saving
# tokenizer_json = word_tokenizer.to_json()
# with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# Adding 1 to store dimensions for words for which no pretrained word embeddings exist
vocab_length = len(word_tokenizer.word_index) + 1
print('vocab_length: ',vocab_length)

# Padding all reviews to fixed length 100
from keras.utils import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



# Import 100 dimension Load GloVe word embeddings and create an Embeddings Dictionary
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

# Create Embedding Matrix having 100 columns
embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
print('embedding_matrix.shape:')
print(embedding_matrix.shape)

# import pickle
# with open('embedding_matrix.pkl', 'wb') as fp:
#     pickle.dump(embedding_matrix, fp)
#     print('embedding_matrix saved successfully to file')
# np.save('X_train.npy', X_train)
# np.save('X_test.npy', X_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)
#
# X_train = np.load("X_train.npy")
# X_test = np.load("X_test.npy")
# y_train = np.load("y_train.npy")
# y_test = np.load("y_test.npy")
# with open('embedding_matrix.pkl', 'rb') as fp:
#     embedding_matrix = pickle.load(fp)


from keras.models import Sequential,Model
from keras.layers import Activation, Dropout ,Dense, LSTM, Embedding,Dropout,SpatialDropout1D,MaxPooling1D,GRU,BatchNormalization
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from keras.layers import Input,Bidirectional,GlobalAveragePooling1D,concatenate,LeakyReLU
from keras import regularizers
from keras import backend as K

# Neural Network architecture
filters = 16
kernel_size = 3
lstm_units = 3
model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters, kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(lstm_units,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters, kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(lstm_units,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(filters, kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(lstm_units,dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(4,activation='softmax'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

# Model compiling
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
print(model.summary())

# Model training
cnn_model_history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Predictions on the Test Set
score = model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
y_pred = model.predict(X_test)
confusion_matrix = tf.math.confusion_matrix(y_test, y_pred)
for i in range(len(y_test)):
    print(y_test[i],y_pred[i])
print(confusion_matrix)

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