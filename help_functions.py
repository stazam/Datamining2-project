import streamlit as st
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


text = "The focus of this tutorial is to learn how to make word clouds that don’t suck on a visual and content-based level. There are tons a free tools online for non-programmers to make word clouds, but visualizing your data within your python pipeline as you iterate parameters during an NLP project can be a powerful way to evaluate the results of your decisions, and present your process to non-technical stakeholders. Having a visual design background, I find the out-of-the-box word cloud package to be sort-of horrendous, so I will offer a few tweaks that will make your word clouds not suck."


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



