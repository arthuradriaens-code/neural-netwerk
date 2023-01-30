'''
json must be in form
[
{ is_sarcastic : 1,  headline :  thirtysomething scientists unveil doomsday clock of hair loss ,  article_link :  https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205 },
...
{ is_sarcastic : 1,  headline :  thirtysomething scientists unveil doomsday clock of hair loss ,  article_link :  https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205 }
]
'''
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
import numpy as np

with open("archive/modified.json",'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Let's Tokenize
vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
num_epochs = 30

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,maxlen=max_length,
                                padding=padding_type,truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length,
                                padding=padding_type,truncating=trunc_type)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(np.array(training_padded),np.array(training_labels),epochs=num_epochs,validation_data=(np.array(testing_padded),np.array(testing_labels)))

sentence = ['hotel manners: towel on floor means to wash, towel in bag means to steal.']
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences,maxlen=max_length,
                                padding=padding_type,truncating=trunc_type)
print(model.predict(padded))
