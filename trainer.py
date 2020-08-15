import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding


print('-> Files and Libraries Imported \n')


def load_dicts():
    start=time.time()
    num_to_vec  = pkl.load(open('num_to_vec.pkl','rb'))
    end=time.time()
    print('-> All Dictionaries Loaded in ',round(end-start,3),' Secs \n')
    return num_to_vec


def load_embd_matrix(num_to_vec):
    embd_matrix=np.zeros((400001,200))
    for i in range(400001):
        embd_matrix[i]=num_to_vec[i]
    return embd_matrix


def load_train_data():
    start=time.time()
    X = np.load('train_review.npy')
    Y = np.load('train_rating.npy').reshape(-1,1)
    end=time.time()
    print('-> Training Data Loaded in ',round(end-start,3),' Secs \n')
    return X,Y


def load_test_data():
    start=time.time()
    X = np.load('test_review.npy')
    Y = np.load('test_rating.npy').reshape(-1,1)
    end=time.time()
    print('-> Test Data Loaded in ',round(end-start,3),' Secs \n')
    return X,Y


def model_def(embd_matrix,x_train,y_train,x_test,y_test):
    start=time.time()
    model = Sequential()
    model.add(Embedding(400001,200,weights=[embd_matrix],input_length=100,trainable=False,mask_zero=True))
    model.add(LSTM(1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())
    end=time.time() 
    print('-> Model Defined in ',round(end-start,3),' Secs \n')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def main():
    num_to_vec=load_dicts()
    embd_matrix=load_embd_matrix(num_to_vec)
    x_train,y_train=load_train_data()
    x_test,y_test=load_test_data()
    model=model_def(embd_matrix,x_train,y_train,x_test,y_test)


main()
