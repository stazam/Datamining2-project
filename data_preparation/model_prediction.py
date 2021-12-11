from data_preparation.data_cleaning import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization,GlobalAveragePooling1D, Flatten, Dropout
import tensorflow as tf


def create_model(load = True, save_model = True):

    if load:
        return load_model(r'C:\Users\zamec\Datamining2-project\model\model.h5')

    model = tf.keras.Sequential()

    model.add(Embedding(vocab_size,240, input_length = max_length))
    model.add(Bidirectional(keras.layers.LSTM(150, return_sequences=True)))
    model.add(Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
    model.add(Bidirectional(keras.layers.LSTM(32, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(3,activation = 'softmax'))


    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
    model.summary()

    if save_model:
        model.save(r'C:\Users\zamec\Datamining2-project\model\model.h5')
    
    return model

def crete_graph(x):

    x = x[0].tolist()
    #pie, ax = plt.subplots(figsize=[10,6])
    fig1 = plt.figure(figsize=(12,7))
    labels = ['science','sport','travel']
    plt.pie(x, autopct="%.1f%%", explode=[0.05]*3, labels=labels, pctdistance=0.5, shadow=True)
    plt.title("Results", fontsize=20)
    plt.show()
    
    result = labels[x.index(max(x))]
    return fig1, result


def main():
    
    #create model
    savedModel =  create_model()
    
    #add some text which will be prepared for prediction
    sentence = [input("Add input as a string: ")]
    sentence = predict_preparation(sentence)
    print(sentence)
    #make prediction and graph
    prediction = savedModel.predict(sentence)
    _, result = crete_graph(prediction)  
    print(result)

if __name__ == "__main__":
    main()

  