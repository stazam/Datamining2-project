from data_preparation.scraper import *
from data_preparation.scraper import *
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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