import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from help_functions import *

st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center; color: #333F4F;'>Klasifikácia textu</h1>", unsafe_allow_html=True)

st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)
with col1:
    user_input = st.text_area("Vložte svoj text sem (slovensky, česky alebo anglicky)", 'Add your text here',height=200)

with col2:
    file_input = st.file_uploader("Vložte textový súbor sem (slovenský, český alebo anglický text)", type = ['.txt'])

# text_file = open(file_input, "r")
# print(text_file)

st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h2 style='text-align: center; color: #333F4F;'>Wordcloud pre zadaný text</h2>", unsafe_allow_html=True)    
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(user_input)
    fig = plt.figure(figsize= (12,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    st.pyplot(fig)

with col2:
    st.markdown("<h2 style='text-align: center; color: #333F4F;'>Frekvencia výskytu slov v texte</h2>", unsafe_allow_html=True)
    df = text_count(user_input)
    
    fig1 = plt.figure(figsize=(12,7))
    plt.bar(df['word'].head(20),df['count'].head(20))
    plt.xticks(rotation=50)
    # plt.xlabel("Country of Origin")
    plt.ylabel("Frequency")
    plt.show()

    st.pyplot(fig1)

