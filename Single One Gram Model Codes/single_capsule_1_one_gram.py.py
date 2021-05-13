import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from functools import partial
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.keras import activations, regularizers, initializers, constraints, engine
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.layers import Layer, deserialize, Conv1D
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops


data = pd.read_csv("1000_tweets_per_user_train.csv")
print(data.head(5))


#data.rename(columns = {'Text':'text', 'authors':'label'}, inplace = True) 
data.rename(columns = {'0':'text', '1': 'label'}, inplace = True)
print(data.info())


print(data['label'].value_counts())

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Flatten, Dense,AveragePooling1D,GRU,Convolution1D, MaxPooling1D, AveragePooling1D,Embedding,Input, Dense, Dropout, Flatten



import pandas as pd
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.33, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
print(data.shape)
#print(dataset.shape)


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


# train
print('Train and Test size = ', train.shape, test.shape)
X = train[train.columns[0]]
y = train[train.columns[1:]]
print(X.shape)
print(y.shape)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



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



import string
maxlen = 140

def create_vocab_set2():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'] + [' '] + list(string.whitespace))

    print('alphabet', alphabet)
    alphabets = []
    for i in range(0, len(alphabet)):
        alphabets.append(alphabet[i])


    vocab_size = len(alphabets)
    check = set(alphabets)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabets):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check

vocab, reverse_vocab, vocab_size, check = create_vocab_set2()

print('vocab size: ', vocab_size)
print(vocab)

def encode_data2(x, maxlen, vocab, vocab_size, check):              # For character embedding use this function instead of encode_data

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        chars2 = list(sent.lower())
        for i,c in enumerate(chars2):
            if counter >= maxlen:
                pass
            else:
                if c in check:
                    counter += 1
                    ix = vocab[c]
                    input_data[dix,counter-1,:] = ix

    return input_data


'''
def encode_data(x, maxlen, vocab, vocab_size, check):

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower())
        two = list(sent.lower())
	for i in range(len(sent.lower())):
 	    chars.append(two[i]+two[i+1])
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data

'''

def encode_data(x, maxlen, vocab, vocab_size, check):

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower())
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


X_train = encode_data(X_train, maxlen, vocab, vocab_size, check)

X_test= encode_data(X_test, maxlen, vocab, vocab_size, check)

print('Before ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(X_train,y_train, test_size = 0.25, random_state = 0)

print('After ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.3
rate_drop_dense = 0.3

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


from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)






def model2(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,cat_output):                                                  # For Character Embedding use this model instead of above model
    d = 300                                                             #Embedding Size
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')
    conv = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0], activation='relu',input_shape=(maxlen, vocab_size))(inputs)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0],
                           activation='relu')(conv)

    
    conv6= Capsule(num_capsule=1 ,dim_capsule=72, routings=1,share_weights=True)(conv1)
    conv = Flatten()(conv1)

    pred = Dense(cat_output, activation='softmax', name='output')(conv)

    model = Model(inputs=inputs, outputs=pred)

    return model




print('Last ', X_train.shape,y_train.shape,X_test.shape,y_test.shape)
nb_filter = 500
dense_outputs = 256
filter_kernels = [3,2,3,4,5]
cat_output = 50
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model = model2(filter_kernels, dense_outputs,maxlen, vocab_size, nb_filter, cat_output)
print(model.summary())

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

#print(result)
#print(predicted)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(result,predicted)
print(cm)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('Confusion Matrix :')
print(cm)
print('Accuracy Score :',accuracy_score(result, predicted))
print('Report : ')
print(classification_report(result, predicted, digits=6))

















