import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from help_functions import *
from data_preparation.model_prediction import *
from data_preparation.data_cleaning import *
from io import StringIO 

st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center; color: #333F4F;'>Text classification</h1>", unsafe_allow_html=True)

st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)
with col1:
    user_input = st.text_area("You can add copy of your text here (in english)", '',height=20)
    text = user_input.strip().rstrip() 

with col2:
    file_input = st.file_uploader("You can drag and grop your text files here (files should be in english)", type = ['.txt'])
    if not file_input is None: 
       
        stringio = StringIO(file_input.getvalue().decode("utf-8"))
        data = stringio.read().strip().rstrip()
        
        if  len(data) != 0:
            text = text + ' ' + data   

st.markdown("")
st.markdown("")

if len(text) != 0:

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h2 style='text-align: center; color: #333F4F;'>Wordcloud pre zadaný text</h2>", unsafe_allow_html=True)    
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
        fig = plt.figure(figsize= (12,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        st.pyplot(fig)

    with col2:
        st.markdown("<h2 style='text-align: center; color: #333F4F;'>Frekvencia výskytu slov v texte</h2>", unsafe_allow_html=True)
        df = text_count(text)
        
        fig1 = plt.figure(figsize=(12,7))
        plt.bar(df['word'].head(20),df['count'].head(20))
        plt.xticks(rotation=50)
        # plt.xlabel("Country of Origin")
        plt.ylabel("Frequency")
        plt.show()

        st.pyplot(fig1)


    savedModel =  create_model()
    sentence = predict_preparation([text])
    prediction = savedModel.predict(sentence)

    graph, result = crete_graph(prediction)

    st.write(result)
    st.pyplot(graph)  