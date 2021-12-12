import streamlit as st
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from tensorflow.python.keras.backend import update
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from help_functions import *
from data_preparation.model_prediction import *
from data_preparation.data_cleaning import *
from io import StringIO, BytesIO 
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide')

# image = Image.open(r'C:\Users\zamec\Datamining2-project\text_classification.png')
# st.image(image, caption='Sunrise by the mountains')

col1, col2, col3 = st.columns([1.1,1,1])
with col2:
    st.header('Text classification')

st.markdown("")
st.markdown("")

col1, col2 = st.columns(2)
with col1:
    user_input = st.text_area("You can add copy of your text here (in english)", '',height=20)
    stats = st.checkbox('Certainty of model prediction', value=False)
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

#set up sidebar
st.sidebar.header('User input features')
st.sidebar.markdown("")
#multiselect choice of graphs
graphs = st.sidebar.multiselect('Which graphs would you like to dispaly',options = ['WordCloud','Barchart'], default=None)        

if 'WordCloud' in graphs:
    init_val_wc = round(len(text.split()) /2 )
    max_val_wc = len(text.split())
    num_words_cloud = st.sidebar.select_slider( 'Number of dispalyed words in word cloud', options=list(range(max_val_wc)), value = init_val_wc)

if 'Barchart' in graphs:
    df_bar = text_count(text)

    max_val_bar = df_bar.shape[0]
    init_val_bar = round(max_val_bar / 2)
    num_words_bar = st.sidebar.select_slider( 'Number of dispalyed words in barchart', options=list(range(max_val_bar)), value = init_val_bar)


st.sidebar.markdown("")
st.sidebar.markdown("")
my_model = st.sidebar.checkbox('Train your own model', value=False) 
if my_model:
    file_input = st.sidebar.file_uploader("Add text files for training NN", type = ['.txt'],accept_multiple_files=True) 
    num_epochs = st.sidebar.select_slider( 'Number of epochs to train NN', options=list(range(100)), value = 3)

    if len(file_input) == 3: 
        train_new_model(file_input, num_epochs)
        


if len(text) != 0:

    if my_model:
        savedModel =  load_my_model(which_model='own')
        sentence = predict_preparation([text], own = True)
    else:
        savedModel =  load_my_model(which_model='pretrained')
        sentence = predict_preparation([text])

    prediction = savedModel.predict(sentence)

    colors = ['#00cc96','#636efa','#ef553b']
    df, result, ind = get_result(prediction)

    color = colors[ind]
    st.markdown("<h2 style='text-align: center; color: #31333f;'>Your article has been predicted as <span style='color:" + color + "'>" + result + "</span> article </h2>", unsafe_allow_html=True) 
    if stats:
        col1, col2, col3 = st.columns([1.1,0.1,1.2])
        with col1:
            fig = go.Figure(data=[go.Table(columnwidth = [40,40],header=dict(values=['Category', 'Prediction'], align =  ['center','center']),
                 cells=dict(values=[df.Category,[str(x) + '%' for x in round(100 * df.Prediction,3)]]))])
            fig.update_layout(legend={"x" : 0.5, "y" : 1})
            st.plotly_chart(fig, use_container_width=True)           
        
        with col2:
            st.markdown('')
        
        with col3:
            fig1 = px.pie(df, values='Prediction', names='Category')  
            st.plotly_chart(fig1, use_container_width=True)  


    if 'Barchart' in graphs:
        st.markdown("""<hr style="height:0.5px;width:1050px;border:none;color:#D2D2D2;background-color:#D2D2D2;margin-top:0px; margin-bottom:0px" /> """, unsafe_allow_html=True)

        col1,col2,col3 = st.columns([0.05,1,0.3])
        with col1:
            st.markdown('')

        with col2:    
            st.markdown("<h2 style='text-align: center; color: #333F4F; margin-left:200px'>Barchart</h2>", unsafe_allow_html=True)
            
            fig = px.bar(df_bar.head(num_words_bar),x = 'word',y="count", color="word", title="Long-Form Input") 
            fig.update_layout(
                autosize=False,
                width=1000,
                height=500,
                showlegend = False,
                xaxis_title=None,
                yaxis_title = 'Number of words',
                title = None
                )           
            st.plotly_chart(fig, use_container_width=False)
        
        with col3:
            st.markdown('')
        
    if 'WordCloud' in graphs:
        st.markdown("""<hr style="height:0.5px;width:1050px;border:none;color:#D2D2D2;background-color:#D2D2D2;margin-top:0px; margin-bottom:0px" /> """, unsafe_allow_html=True)

        col1,col2,col3 = st.columns([0.25,1,0.2])
        with col1:
            st.markdown('')

        with col2:
            st.markdown("<h2 style='text-align: center; color: #333F4F;'>Wordcloud</h2>", unsafe_allow_html=True) 
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('') 
            
            wordcloud = WordCloud(max_words=num_words_cloud , background_color="white", width = 600, height= 300).generate(text)
            fig, ax = plt.subplots(figsize=(8, 7))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig)

        with col3:
            st.markdown('')

    

    #     savedModel =  create_model(update = True)

    #     sentence = predict_preparation([text])
    #     prediction = savedModel.predict(sentence) 
