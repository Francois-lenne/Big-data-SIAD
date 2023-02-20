import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import os
import sys
import re
import string
import re
import spacy
from sympy import symbols, Eq, solve
from geotext import GeoText
import tqdm
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
import joblib
from textblob import TextBlob
nlp = spacy.load("en_core_web_sm")
stopwords = spacy.lang.en.stop_words.STOP_WORDS

def precision(text):
        rep = '0'
        reg1 = r'(,)'
        p = re.compile(reg1)
        check = re.search(reg1,text)
        if check is not None:
            rep = '1'
        return rep

def city(text):
    str(text).capitalize()
    places = GeoText(text)
    if len(places.cities) == 0:
        cities = 0
    else:
        cities = 1
    return cities

def country(text):
    str(text).capitalize()
    places = GeoText(text)
    if len(places.countries) == 0:
        countries = 0
    else:
        countries = 1
    return countries

def recupHashtagBinaire(text):
    rep = '0'
    reg = r"#(\w+)"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
        rep = '1'
    return rep

def recupNameBinaire(text):
    rep = '0'
    reg = r"@(\w+)"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
        rep = '1'
    return rep

def recupDate(text):
    reg = r"([A-Za-z]{3})\s(\d{1,2}),\s(\d{4})"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
        ge = p.findall(text)
        return ge

def recupDateBinaire(text):
    rep = '0'
    reg = r"([A-Za-z]{3})\s(\d{1,2}),\s(\d{4})"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
        rep = '1'
    return rep

def getCheminBinaire(text):
    rep = '0'
    reg1 = r'(https?:\/\/[^\s]+)'
    p = re.compile(reg1)
    check = re.search(reg1,text)
    if check is not None:
        rep = '1'
    return rep

def getLocation(text):
    global nlp
    save = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            save.append(ent.text)
    return save

def getLocationBinaire(text):
    rep = "0"
    global nlp
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            rep = "1"
    return rep

def getSubjectivity(text):
    subj = TextBlob(text).sentiment.subjectivity
    if subj < 0:
        score = 2
    elif subj == 0:
        score = 0
    else:
        score = 1
    return score
    
def getPolarity(text):
    polar = TextBlob(text).sentiment.polarity
    if polar < 0:
        score = 2
    elif polar == 0:
        score = 0
    else:
        score = 1
    return score

def preprocessing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text) # retrait liens
    text = re.sub(r'&amp|& amp', '&', text) # remplacement &amp par &
    text = re.sub(r'#\S+', '', text) # retrait hashtags
    text = re.sub(r'@\S+', '', text) # retrait arobases
    text = re.sub('[Ã©Ã¨Ãª]', "e", text) # retrait accents sur le e
    text = re.sub("Ã´", "o", text) # retrait accents sur le o
    text = re.sub("[.,;:!?]", " ", text) # retrait ponctuation
    text = re.sub("[|{}\[\]()Â«Â»/]", " ", text) # retrait parenthÃ¨ses, crochets, guillemets, slashs...
    text = re.sub("[â€œâ€]", " ", text) # retrait guillemets (autre forme)
    text = re.sub("'", " ", text) # retrait apostrophes
    text = re.sub("â€™", " ", text) # retrait apostrophes (autre forme)
    text = re.sub('"', " ", text) # retrait quotes
    text = re.sub('[+-]', " ", text) # retrait + et -
    text = re.sub('[=*/]', " ", text) # retrait opÃ©rateurs
    text = re.sub("Â°", "", text) # retrait symbole Â°
    text = re.sub("[â‚¬%$Â£]", "", text) # retrait symboles devises
    text = re.sub('\r\n', " ", text) # retrait retour charriot/retour Ã  la ligne
    text = re.sub('\n', " ", text) # retrait retour Ã  la ligne
    text = re.sub('\s+', " ", text) # retrait espaces en trop
    text = text.rstrip(" ") # retrait espaces Ã  droite
    text = text.lstrip(" ") # retrait espaces Ã  gauche
    return text

def Lemmatization(train,texts):
    pbar = tqdm.tqdm(total=train.shape[0])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc: 
            new_text.append(token.lemma_)
            final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

def tokenize(text):
    text_split = [word for word in text.split()]
    return text_split

def remove_stopwords(text):
    global stopwords
    text = [word for word in text if (len(word) > 2) and (word not in stopwords)]
    return text


def DataPreparation(train, sample_type):

    train['location'] = train['location'].astype(str)

    train['city'] = train['location'].apply(city)
    print(train)
    print('phase 1')
    train['country'] = train['location'].apply(country)
    train['precision'] = train['location'].apply(precision)
    train['hashtags_b'] = train['text'].apply(recupHashtagBinaire)
    train['names_b'] = train['text'].apply(recupNameBinaire)
    train['dates'] = train['text'].apply(recupDate)
    train['dates_b'] = train['text'].apply(recupDateBinaire)
    train['locations'] = train['text'].apply(getLocation)
    train['locations_b'] = train['text'].apply(getLocationBinaire)
    train['rt_path_b'] = train['text'].apply(getCheminBinaire)
    train['Subjectivity'] = train['text'].apply(getSubjectivity)
    train['Polarity'] = train['text'].apply(getPolarity)
    print(train)
    print('phase 2')
    train['text_CLEAN'] = train['text'].apply(lambda x: preprocessing(x))
    train['text_CLEAN_LMT'] = Lemmatization(train,train['text_CLEAN'])
    train['text_CLEAN_LMT_TOKEN'] = train['text_CLEAN_LMT'].apply(lambda x: tokenize(x))
    print(train)
    print('phase 3')

    global nlp
    global stopwords
    stopwords = [word.lower() for word in stopwords]

    train['text_CLEAN_LMT_TOKEN_WSW'] = train['text_CLEAN_LMT_TOKEN'].apply(lambda x: remove_stopwords(x))
    print(train)
    print('phase 4')
    data_words_train = train['text_CLEAN_LMT_TOKEN_WSW'].values.tolist()
    bigram = gensim.models.Phrases(data_words_train, min_count=5, threshold=0,scoring = 'npmi')
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    print(train)
    print('phase 5')

    def make_bigrams(texts):
        return [bigram_mod[text] for text in texts]

    train['text_CLEAN_LMT_TOKEN_WSW_BIGRAMS'] = make_bigrams(data_words_train)
    
    def detokenize_text(txt):
        txt = ' '.join([word for word in txt])
        return txt

    train['text_CLEAN_LMT_WSW_BIGRAMS'] = train['text_CLEAN_LMT_TOKEN_WSW_BIGRAMS'].apply(lambda x: detokenize_text(x))
    print(train)
    print('phase 6')
    # Fitter le vectorizer en amont sur les donnÃ©es d'entraÃ®nement

    
    vectorizer = joblib.load('vectorizer.gz')
   
    vectors = vectorizer.transform(train['text_CLEAN_LMT_WSW_BIGRAMS'].astype('U'))
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    train_word_features = pd.DataFrame(dense, columns=feature_names)

    print(train)
    print('phase 7')
    train_variables_extracted = train[['hashtags_b','names_b','rt_path_b','locations_b', 'Subjectivity', 'Polarity']]
    train_location = train[['city','country','precision']]

    print("7.1", train_variables_extracted, train_location)

     # final_train = pd.concat([train_word_features, # variables crÃ©Ã©es Ã  l'aide des tweets (features TF IDF)
                            # train_location, # variables crÃ©Ã©es Ã  partir des localisations
                            # train_variables_extracted], # variables crÃ©Ã©es via l'extraction de donnÃ©es dans les tweets
                            # axis='columns')


    final_train = train[['hashtags_b','names_b','rt_path_b','locations_b', 'Subjectivity', 'Polarity','city','country','precision']]
    print("7.2", final_train)
    X_train = final_train.reset_index(drop=True)

    print(train)
    print('phase 8')

    if sample_type == True:
        if 'target' in train:
            y_train = train['target']
            return X_train, y_train
    
    else:
        return X_train

def vectorizerCreate(train):
    train['location'] = train['location'].astype(str)
    train['city'] = train['location'].apply(city)
    train['country'] = train['location'].apply(country)
    train['precision'] = train['location'].apply(precision)
    train['hashtags_b'] = train['text'].apply(recupHashtagBinaire)
    train['names_b'] = train['text'].apply(recupNameBinaire)
    train['dates'] = train['text'].apply(recupDate)
    train['dates_b'] = train['text'].apply(recupDateBinaire)
    train['locations'] = train['text'].apply(getLocation)
    train['locations_b'] = train['text'].apply(getLocationBinaire)
    train['rt_path_b'] = train['text'].apply(getCheminBinaire)
    train['Subjectivity'] = train['text'].apply(getSubjectivity)
    train['Polarity'] = train['text'].apply(getPolarity)
    train['text_CLEAN'] = train['text'].apply(lambda x: preprocessing(x))
    train['text_CLEAN_LMT'] = Lemmatization(train,train['text_CLEAN'])
    train['text_CLEAN_LMT_TOKEN'] = train['text_CLEAN_LMT'].apply(lambda x: tokenize(x))

    global nlp
    global stopwords
    stopwords = [word.lower() for word in stopwords]

    train['text_CLEAN_LMT_TOKEN_WSW'] = train['text_CLEAN_LMT_TOKEN'].apply(lambda x: remove_stopwords(x))

    data_words_train = train['text_CLEAN_LMT_TOKEN_WSW'].values.tolist()
    bigram = gensim.models.Phrases(data_words_train, min_count=5, threshold=0,scoring = 'npmi')
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    def make_bigrams(texts):
        return [bigram_mod[text] for text in texts]

    train['text_CLEAN_LMT_TOKEN_WSW_BIGRAMS'] = make_bigrams(data_words_train)

    def detokenize_text(txt):
        txt = ' '.join([word for word in txt])
        return txt

    train['text_CLEAN_LMT_WSW_BIGRAMS'] = train['text_CLEAN_LMT_TOKEN_WSW_BIGRAMS'].apply(lambda x: detokenize_text(x))

    vectorizer = TfidfVectorizer()
    vectorizer_fit = vectorizer.fit(train['text_CLEAN_LMT_WSW_BIGRAMS'].astype('U'))
    joblib.dump(vectorizer_fit,'vectorizer.gz')