import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
        'I love my dog',
        'I love my cat',
        'You love my Dog!' #uppercase will be lowered and ! stripped
        'Do you think my dog is amazing?'
        ]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>") #maximum number of words
tokenizer.fit_on_texts(sentences) #lets tokenizer do it's job (fit on array)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences) #Array of sentences to array of tokes

test_data = [
        'I really love my dog',
        'my dog loves my manatee'
        ]

test_seq = tokenizer.texts_to_sequences(test_data)

padded = pad_sequences(sequences)
print(sequences)
print(padded)
