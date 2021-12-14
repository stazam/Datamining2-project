from data_preparation.data_cleaning import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization,GlobalAveragePooling1D, Flatten, Dropout, LSTMCell
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st


def create_model():

    model = tf.keras.Sequential()

    model.add(Embedding(vocab_size,240, input_length = max_length))
    model.add(Bidirectional(keras.layers.LSTM(150, return_sequences=True)))
    model.add(Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
    model.add(Bidirectional(keras.layers.LSTM(32, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(3,activation = 'softmax'))


    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
    model.summary()
    
    return model


@st.cache(suppress_st_warning=True)
def train_new_model(file_input, num_epochs):

    X_train_padded, X_test_padded, y_train, y_test = preprocess_input_to_model(file_input)

    model = create_model()
    my_bar = st.sidebar.progress(0)
    percent = round(100 / num_epochs)
    for num in range(num_epochs):
        history = model.fit(X_train_padded, y_train, epochs=1, validation_data=(X_test_padded, y_test)) 
        my_bar.progress((num + 1) * percent)

    model.save(r'C:\Users\zamec\Datamining2-project\model\model_new.h5')
    print(history.history)
    acc = round(history.history['val_accuracy'][0],2) * 100
    loss = round(history.history['val_loss'][0],2)
    st.sidebar.write('The accuracy of a model on testing set is: ',acc)
    st.sidebar.write('With the loss value: ',loss)



def load_my_model(which_model:str):

    if which_model == 'pretrained':
        return load_model(r'C:\Users\zamec\Datamining2-project\model\model.h5')
    elif which_model == 'own':
        return load_model(r'C:\Users\zamec\Datamining2-project\model\model_new.h5')  

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


def get_result(x):

    x = x[0].tolist()
    #pie, ax = plt.subplots(figsize=[10,6])
    categories = ['science','sport','travel']

    ind = x.index(max(x))    
    result = categories[ind]
    df = pd.DataFrame(list(zip(categories, x)), columns =['Category', 'Prediction'])
    
    return df, result, ind    


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

  