# Tokenisation

I love my dog $\rightarrow$ 001 002 003 004
I love my cat  $\rightarrow$ 001 002 003 005

{'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}

my most common.

((4, 2, 1, 3), (4, 2, 1, 6), (5, 2, 1, 3, 7, 5, 8, 1, 3, 9, 10)):

(actually square brackets but obs doesnt like those)

first sentence:
(4, 2, 1, 3)

Token used for dog should be same when training network and later on when testing!

```python
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
```

OOV allows you to specifiy a token type for Out Of Vocabulary.

# padding

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(sequences)

print(sequences)
print(padded)
```

gives

((5, 3, 2, 4), (5, 3, 2, 7), (6, 3, 2, 4, 8, 6, 9, 2, 4, 10, 11))

((0  0  0  0  0  0  0  5  3  2  4)

 ( 0  0  0  0  0  0  0  5  3  2  7)

 ( 6  3  2  4  8  6  9  2  4 10 11))
 
(get equally long senctences)

if you want paddingat end: padding='post' and also a maxlen=5 parapeter can be given to pad_sequences this ofc gives truncating, that can also be given 'post' argument.

# Embedding

```python
model = keras.Sequential([
	keras.layers.Embedding(
					   vocab_size,embedding_dim,input_length=max_length
					   ),
    keras.layers.GlobalAveragePooling1D(),                         
    keras.layers.Dense(24,activation='relu'),                      
    keras.layers.Dense(1,activation='sigmoid')                     
    ])
```

embedding is like vectors e.g good is [1,0], bad is [-1,0] and meh is [-1/\sqrt{2},$1/\sqrt{2}]

vocab_size vectors with embedding_dim dimensions.
