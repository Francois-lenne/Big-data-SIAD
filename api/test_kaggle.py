import pandas as pd
import csv
import re 
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


train = pd.read_csv("https://raw.githubusercontent.com/Francois-lenne/Big-data-SIAD/main/data/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Francois-lenne/Big-data-SIAD/main/data/test.csv")


print(train)

train = train.drop(['id', 'keyword', 'location'], axis = 1)
test = test.drop(['id', 'keyword', 'location'], axis = 1)

train_data = train

from sklearn.model_selection import train_test_split

# splitting the training and testing part from the data
X_temp, X_test, y_temp, y_test = train_test_split(train_data['text'], train_data['target'], test_size=0.2, random_state=0)

train_X = X_temp
train_y = y_temp
test_x = X_test
test_y = y_test

train_X.shape, train_y.shape, test_x.shape , test_y.shape

label = preprocessing.LabelEncoder()
y = label.fit_transform(train_X)
y = to_categorical(train_y)
print(train_y[:5])

m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

import nltk
nltk.download("popular")
from nltk.tokenize import word_tokenize

from bert import tokenization
from bert.tokenization.bert_tokenization import FullTokenizer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(2, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

max_len = 250
train_input = bert_encode(train_X.values, tokenizer, max_len=max_len)
test_input = bert_encode(test_x.values, tokenizer, max_len=max_len)
train_labels = train_y

labels = label.classes_
print(labels)

model = build_model(bert_layer, max_len=max_len)
model.summary()

label = preprocessing.LabelEncoder()
y = label.fit_transform(train_labels)
y = to_categorical(y)
print(y[:5])

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_sh = model.fit(
    train_input, y,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=32,
    verbose=1
)

prediction = model.predict(test_input)

prediction

prediction.shape

pred = np.argmax(prediction, axis = 1)

pred.shape

pred

from sklearn.metrics import accuracy_score
accuracy_score(test_y, pred)