from tensorflow.python.eager.context import num_gpus
from data_preparation.data_cleaning import *
from data_preparation.model_prediction import *
import streamlit as st
from io import StringIO
from sklearn.model_selection import train_test_split
import tensorflow as tf

#rozdelit na funkcie a potom pridat do model_prediciton
num_epochs = 5
file_input = st.file_uploader("You can drag and grop your text files here (files should be in english)", type = ['.txt'],accept_multiple_files=True)
# if len(file_input) == 2:
#     for file in file_input:
#         file.name
if len(file_input) == 3: 
    
    labels = []
    data = []
    for i,file in enumerate(file_input):
        stringio = StringIO(file.getvalue().decode("utf-8"))
        
        text = []
        length = 1
        while length > 0:
            line = stringio.readline()
            if len(line) > 0:
                text.append(line)
            length = len(line)

        text = dict([(x.split(':')[0],[x.split(':')[1]]) for x in text])
        data_temp = preprocess_articles(text)    
        data = data + data_temp 
        labels = labels + [i] * len(data_temp)

    num_class = len(set(labels))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    tokenizer = create_tokenizer(X_train)    
    X_train_padded = predict_preparation(X_train, own = True)
    X_test_padded = predict_preparation(X_test, own = True)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_class)

    model = create_model()
    
    my_bar = st.progress(0)
    percent = round(100 / num_epochs)
    for num in range(num_epochs):
        history = model.fit(X_train_padded, y_train, epochs=1, validation_data=(X_test_padded, y_test)) 
        my_bar.progress(num + percent)

    acc = history.history['val_accuracy'][num_epochs - 1]
    loss = history.history['val_loss'][num_epochs - 1]
    st.wtite('The accuracy of a model on testing set is: ',acc)
    st.wtite('With the loss value: ',loss)
    

    #return model
    
# def parse_input(articles) -> list:

  