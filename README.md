# Text classificator

This repo is intended for final project of course M9DM2 Data mining 2.

**Authors**: Stanislav Zámečník, Vojtěch Šindlář.

**Description of project**: the web application is intended for classifying articles into three categories: **sport, travel, science** (articles are supposed to be in english). 

**Features**: 
1. Categorizing copy of any text article. 
2. Possibility to add .txt file for categorization. 
3. It's also possible to use combination of copy of text and text from .txt file. 
4. Use of webscraper to obtain own data from page https://www.dailymail.co.uk/home/index.html 
5. Possibility to train your own model.


**Folders**: 
- data - .npy files containing n articles from dailymail.co page for pretraining
- data_preparation - contains scripts for web-scraping, data cleaning and modelling
- model - contain saved weights ofr pretrained model in format .h5 
- files_for_deployment - necessary files for heroku deployment
- test_files - files for showing the functionality of web application.
         

**How to run app**: streamlit run app.py
