from data_preparation.scraper import *
from data_preparation.scraper import *
import pickle
import tensorflow as tf
from io import StringIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

max_length = 10000
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
vocab_size = 10700

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