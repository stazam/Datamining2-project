from data_preparation.scraper import *
from data_preparation.scraper import *
import pickle
import tensorflow as tf
from io import StringIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st

max_length = 5000
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
vocab_size = 10700

def text_count(text):
    d = dict()
    for line in text.split():
        print(line)
        line = line.strip()
        line = line.lower()
        #line = line.translate(line.maketrans("", "", string.punctuation))
    
        words = line.split(" ")
    
        for word in words:
            if word in d:
                d[word] = d[word] + 1
            else:
                d[word] = 1
    
    for key in list(d.keys()):
        print(key, ":", d[key])

    return pd.DataFrame(list(d.items()), columns = ['word','count']).sort_values('count',ascending=False)


def preprocess_articles(articles) -> list:

  articles_preprocess = []

  for key,value in articles.items():
    list_temp = []
    list_temp.append(key.lower().strip().rstrip())
    for text in value:
      if len(text) > 0:
        list_temp.append(text.lower().strip().rstrip())
    articles_preprocess.append(list_temp)

  articles_preprocess = [' '.join(item) for item in articles_preprocess]

  return articles_preprocess 


def create_tokenizer(x:list):
  
  tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
  tokenizer.fit_on_texts(x)

  with open(r'C:\Users\zamec\Datamining2-project\model\tokenizer_new.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def predict_preparation(text:list, own:bool = False):

  if not own:
    with open(r'C:\Users\zamec\Datamining2-project\model\tokenizer.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
  else:
    with open(r'C:\Users\zamec\Datamining2-project\model\tokenizer_new.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)


  sequences = tokenizer.texts_to_sequences(text)
  return pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


def preprocess_input_to_model(file_input, num_epochs = 5):

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

    create_tokenizer(X_train)    
    X_train_padded = predict_preparation(X_train, own = True)
    X_test_padded = predict_preparation(X_test, own = True)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_class)

    return X_train_padded, X_test_padded, y_train, y_test


def chceck_input(file_input):

    if not len(file_input) == 3:
       return None
        
    parsed_object = [(x.name.split('.')[0],x.name.split('.')[1]) for x in file_input]

    for name, type_file in parsed_object:
        if name in ['sport','travel','science'] and type_file == 'txt':
            continue
        else:
            st.sidebar.info('Add three .txt files with names: sport.txt, travel.txt, science.txt.')
            return None
            break   
    else:
        print('Everything is fine now!')
        file_input_fin = [0,0,0]
        for i,file in enumerate(file_input):
            file_input_fin[['sport','travel','science'].index(parsed_object[i][0])] = file

    return file_input_fin 