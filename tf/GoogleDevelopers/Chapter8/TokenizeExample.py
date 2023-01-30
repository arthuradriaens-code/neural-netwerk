import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
        'I love my dog',
        'I love my cat',
        'You love my Dog!' #uppercase will be lowered and ! stripped
        ]

tokenizer = Tokenizer(num_words=100) #maximum number of words
tokenizer.fit_on_texts(sentences) #lets tokenizer do it's job (fit on array)
word_index = tokenizer.word_index

print(word_index)
