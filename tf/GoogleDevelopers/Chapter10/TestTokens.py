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

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences,padding='post')

print(padded[0])
