import pandas as pd
import csv

"""
Para importar los archivos necesarios para utilizar la biblioteca nltk
se debe instalar la misma con pip, y luego descomentar las líneas
siguiente:

import nltk
nltk.download()

"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
import json


def vectorizer (dicts):

    for i in  dicts:
        dicts[i]['amounts'] = []
        dicts[i]['vector_values'] = []

        text =[]

        for j in dicts[i]['text']:
            text_tokens = tokenize(j)
            text.append(text_tokens)

        text_csv(text)
        words = pd.read_csv("text.csv")

        count = CountVectorizer()
        vect = TfidfVectorizer(max_features=100)

        amounts = count.fit_transform(words.description).toarray()
        values = vect.fit_transform(words.description).toarray()

        dicts[i]['amounts'].append(amounts)
        dicts[i]['vector_values'].append(values)

    return dicts

def tokenize(text):

    response = []

    '''Cogemos los stopwords'''
    _stopwords = stopwords.words('spanish')
    '''Cogemos los signos de puntuacion'''
    no_words = list(punctuation)
    '''Tokenizacion'''
    tokens = word_tokenize(text)
    '''Eliminacion de stopwords'''
    tokens = [element for element in tokens if (element not in _stopwords and element not in no_words)]
    '''Stemming'''
    stems = SnowballStemmer('spanish') 
    tokens_stem = [stems.stem(token) for token in tokens]
    response.append(tokens_stem)

    return response


def text_csv (list):
    with open ('text.csv','w') as csvfile:
        fieldnames = ['description']
        writer = csv.DictWriter(csvfile, fieldnames= fieldnames)
        writer.writeheader()
        for i in list:
            writer.writerow({'description': i})

def take_data():

    with open('data/data.json', 'r+') as file:
        data = json.load(file)

    dict = vectorizer(data)

    return dict


def take_data_reduc ():

    with open('data/data.json', 'r+') as file:
        dicts = json.load(file)
    words = []
    for i in  dicts:
        text =[]
        for j in dicts[i]['text']:
            text_tokens = tokenize(j)
            text.append(text_tokens)

        text_csv(text)
        word = pd.read_csv("text.csv")
        words.append(word)

    return words
