

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.layers import merge
from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations, regularizers, initializers, constraints, engine
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.layers import Layer, deserialize, Conv1D, Embedding
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Flatten, Dense,AveragePooling1D,GRU,Convolution1D, MaxPooling1D, AveragePooling1D,Embedding,Input, Dense, Dropout, Flatten, LSTM

import string
import pandas as pd

data = pd.read_csv('1000_tweets_per_user_train.csv')
data.head()


import pandas as pd
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data['0'], data['1'], test_size=0.33, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
print(data.shape)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)
print('After Changing to Dataframe ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)
X_train=X_train.iloc[:,:].values
X_test=X_test.iloc[:,:].values
y_train=y_train.iloc[:,:].values
y_test=y_test.iloc[:,:].values
print('After Changing to Dataframe ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)

train=np.concatenate((X_train,y_train),axis=1)
test=np.concatenate((X_test,y_test),axis=1)
print('Train and Test size = ', train.shape, test.shape)
np.random.shuffle(train)
np.random.shuffle(test)

train=pd.DataFrame(train)
test=pd.DataFrame(test)
print(train.shape)
print(test.shape)

print('Train and Test size = ', train.shape, test.shape)
X = train[train.columns[0]]
y = train[train.columns[1:]]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

import nltk
nltk.download('wordnet')

import nltk
nltk.download('stopwords')

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
y_train = np_utils.to_categorical(encoded_Y)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = np_utils.to_categorical(encoded_Y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def create_vocab_set2():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'] + [' '] )

    alphabets = []
    for i in range(0, len(alphabet)):
        for j in range(0, len(alphabet)):
            for k in range(0, len(alphabet)):
                alphabets.append(alphabet[i]+alphabet[j]+alphabet[k])

    vocab_size = len(alphabets)
    check = set(alphabets)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabets):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check



maxlen = 140
vocab, reverse_vocab, vocab_size, check =create_vocab_set2()
print('Vocab size: ',vocab_size)

def encode_data2(x, maxlen, vocab, vocab_size, check):              # For character embedding use this function instead of encode_data

    input_data = np.zeros((len(x), maxlen))
    for dix, sent in enumerate(x):
        counter = 0
        chars = list(sent.lower().replace(" ", ""))
        chars2 = []
        for i in range(len(chars)-2):
            chars2.append(chars[i] + chars[i+1]+chars[i+2])
        for i,c in enumerate(chars2):
            if counter >= maxlen:
                pass
            else:
                if c in check:
                    counter += 1
                    ix = vocab[c]
                    input_data[dix,counter-1] = ix

    return input_data

X_train = encode_data2(X_train, maxlen, vocab, vocab_size, check)

X_test= encode_data2(X_test, maxlen, vocab, vocab_size, check)

print('Before ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(X_train,y_train, test_size = 0.25, random_state = 0)
print(X_train.shape, y_train.shape, X_eval.shape, y_eval.shape)

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )

    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def model2(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,cat_output):                                                  # For Character Embedding use this model instead of above model
    d = 300                                                             #Embedding Size
    Embedding_layer = Embedding(vocab_size+1, d, input_length=maxlen)
    inputs = Input(shape=(maxlen,), name='input', dtype='float32')
    embed = Embedding_layer(inputs)
    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0], activation='relu',input_shape=(maxlen, d))(embed)
    conv2 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1], activation='relu',input_shape=(maxlen, d))(embed)
    conv3 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[2], activation='relu',input_shape=(maxlen, d))(embed)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0],
                           activation='relu')(conv1)
    conv2 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1],
                           activation='relu')(conv2)

    conv3 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[2],
                           activation='relu')(conv3)


    conv1 = MaxPooling1D()(conv1)
    conv2 = MaxPooling1D()(conv2)
    conv3 = MaxPooling1D()(conv3)

    conv2 = Capsule(num_capsule=1 ,dim_capsule=72, routings=1,share_weights=True)(conv2)
	
    conv1 = LSTM(100)(conv1)

    conv3 = GRU(100)(conv3)



    conv1 = Flatten()(conv1)
    conv2 = Flatten()(conv2)
    conv3 = Flatten()(conv3)

    conv_comb = tf.keras.layers.Concatenate(axis=1)([conv1, conv2, conv3])
    pred = Dense(cat_output, activation='softmax', name='output')(conv_comb)
    model = Model(inputs=inputs, outputs=pred)

    return model


print('Last ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)

nb_filter = 500
dense_outputs = 256
filter_kernels = [1,4,3,4,5]
cat_output = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model = model2(filter_kernels, dense_outputs,maxlen, vocab_size, nb_filter, cat_output)
model.summary()

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#adam1=Adam(lr=1.00, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train),validation_data=(np.array(X_eval), np.array(y_eval)),batch_size=32,epochs=200, verbose=1,callbacks=[callback])

y_pred = model.predict(np.array(X_test))
y_pred=pd.DataFrame(y_pred)
y_pred=y_pred.eq(y_pred.where(y_pred != 0).max(1), axis=0).astype(int)
y_pred=y_pred.iloc[:,:].values

y_test=pd.DataFrame(y_test)
y_test=y_test.eq(y_test.where(y_test != 0).max(1), axis=0).astype(int)
y_test=y_test.iloc[:,:].values

result=[]
for i in range(0,len(y_test)):
  for j in range(0,len(y_test[0])):
    if(y_test[i][j]==1):
      result.append(j)

predicted=[]
for i in range(0,len(y_pred)):
  for j in range(0,len(y_pred[0])):
    if(y_pred[i][j]==1):
      predicted.append(j)

# print(result)
# print(predicted)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(result,predicted)
cm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Confusion Matrix :')
print(cm)
print('Accuracy Score :',accuracy_score(result, predicted))
print('Report : ')
print(classification_report(result, predicted, digits=6))

X_train.shape
X_eval.shape
X_test.shape

