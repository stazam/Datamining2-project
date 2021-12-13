import streamlit as st
from data_preparation.data_cleaning import *

file_input = st.file_uploader("Add text files for training NN", type = ['.txt'],accept_multiple_files=True) 
num_epochs = st.select_slider( 'Number of epochs to train NN', options=list(range(100)), value = 1)


def chceck_input(file_input):

    parsed_object = [(x.name.split('.')[0],x.name.split('.')[1]) for x in file_input]

    for name, type_file in parsed_object:
        if name in ['science','sport','travel'] and type_file == 'txt':
            continue
        else:
            st.info('Add three .txt files with names: sport.txt, travel.txt, science.txt (in this order).')
            return None
            break   
    else:
        print('Everything is fine now!')
        file_input_fin = [0,0,0]
        for i,file in enumerate(file_input):
            file_input_fin[['science','sport','travel'].index(parsed_object[i][0])] = file

    return file_input_fin              

chceck_input(file_input)