# Ce code est un exemple de pre-traitement sur une autre base de données

# Importation des packages

import string
import pandas as pd
!{sys.executable} -m pip install nltk
import nltk
import re
import sys  
!{sys.executable} -m pip install contractions
import contractions
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
!{sys.executable} -m pip install word2number
from word2number import w2n
!{sys.executable} -m pip install unidecode
import unidecode
from nltk.stem.porter import PorterStemmer
!{sys.executable} -m pip install textblob
from textblob import Word

# Fonctions de traitement texte

def dlDdataset(file):
    df = pd.read_csv(file, sep='\t')
    print(df.columns)
    df_eng = df[df['Definitely English'] == 1]
    df_eng = df_eng.head(10)
    print(df_eng['Tweet'])
    return df_eng

def deleteSpaces(text):
    text_without_space = re.sub(' +', ' ', text)
    return text_without_space

def textToLower(text):
    text_lower = text.lower()
    return text_lower

def transformContraction(text):
    text_transform = contractions.fix(text)
    return text_transform

def deletePunctuation(text):
    punctuation_table = str.maketrans('','',string.punctuation)
    text_without_punctuation = text.translate(punctuation_table)
    return text_without_punctuation

def transformUnicode(text):
    text_unicode = unidecode.unidecode(text)
    return text_unicode

def deleteStopwords(list):
    stop_words = set(stopwords.words('english'))
    filtred = [word for word in list if word not in stop_words]
    return filtred

def uniformisation(list):
    uni_list = []
    for mot in list:
        stemmer = PorterStemmer()
        uni = stemmer.stem(mot)
        uni_list.append(uni)
    return uni_list

def deleteHTTP(list):
    regex = '^http*'
    list_without_http = []
    for mot in list:
        check = re.search(regex,mot)
        if check is None:
            list_without_http.append(mot)
    return list_without_http

def correction(list):
    correction_list = []
    for mot in list:
        word = Word(mot)
        word_correction = word.correct()
        correction_list.append(word_correction)
    return correction_list
    

# Application précdurale des fonctions

def NormalizationDataset(df,param_correction,param_uniforme):
    print(df['Tweet'])
    #On retire les espaces en trop
    df['Tweet'] = df['Tweet'].apply(deleteSpaces)
    #On passe les caractères en minuscule
    df['Tweet'] = df['Tweet'].apply(textToLower)
    #On transforme les contractions
    df['Tweet'] = df['Tweet'].apply(transformContraction)
    #On retire la ponctuation
    df['Tweet'] = df['Tweet'].apply(deletePunctuation)
    #On transforme en unicode
    df['Tweet'] = df['Tweet'].apply(transformUnicode)
    #Tokensization
    df['Tweet'] = df['Tweet'].apply(word_tokenize)
    #Filtre des stopswords
    df['Tweet'] = df['Tweet'].apply(deleteStopwords)
    #On supprime les adresses HTTP
    df['Tweet'] = df['Tweet'].apply(deleteHTTP)
    # On corrige les mots mals orthographiés
    print(df['Tweet'])
    if param_correction == True:
        df['Tweet'] = df['Tweet'].apply(correction)
        print(df['Tweet'])
    #On uniformise les mots de même racine
    if param_uniforme == True:
        df['Tweet'] = df['Tweet'].apply(uniformisation)
    print(df['Tweet'])
    

 # Initialisation de la normalisatio