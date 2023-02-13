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
nlp = spacy.load("en_core_web_sm")

def precision(text):
    rep = '0'
    reg1 = r'(,)'
    p = re.compile(reg1)
    check = re.search(reg1,text)
    if check is not None:
      rep = '1'
    return rep

def city(tweet):
    tweet.capitalize()
    places = GeoText(tweet)
    if len(places.cities) == 0:
        cities = 0
    else:
        cities = 1
    return cities

def country(tweet):
    tweet.capitalize()
    places = GeoText(tweet)
    if len(places.countries) == 0:
        countries = 0
    else:
        countries = 1
    return countries


def recupHashtag(text):
    reg = r"#(\w+)"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
        ge = p.findall(text)
        ge_join = ' '.join(ge)
        return ge_join

def recupHashtagBinaire(text):
    rep = '0'
    reg = r"#(\w+)"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
      rep = '1'
    return rep


def recupName(text):
    reg = r"@(\w+)"
    p = re.compile(reg)
    check = re.search(reg,text)
    if check is not None:
        ge = p.findall(text)
        ge_join = ' '.join(ge)
        return ge_join

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


def getChemin(text):
    reg1 = r'(https?:\/\/[^\s]+)'
    reg2 = r'(https?)://([^:/]+)(?::(\d+))?(/[^?]*)?(\?[^#]*)?(#.*)?'
    p = re.compile(reg1)
    check = re.search(reg1,text)
    if check is not None:
        ge = p.findall(text)
        for val in ge:
            match = re.search(reg2,val)
            if match:
                rt = match.group(4)
            return rt

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


def preprocessing(text):
    text = str(text)

    # Harmonisation - mise en minuscule
    text = text.lower()

    # Gestion des accents et ponctuations
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub("\d+", " ", text) # normalisation nombres
    text = re.sub('[éèê]', "e", text) # retrait accents
    text = re.sub("[.,;:!?]", " ", text)
    text = re.sub("[|{}\[\]()«»/]", " ", text)
    text = re.sub("[“”]", " ", text)
    text = re.sub("'", " ", text)
    text = re.sub("’", " ", text)
    text = re.sub('"', " ", text)
    text = re.sub('[+-]', " ", text)
    text = re.sub('[=*/]', " ", text)
    text = re.sub("ô", "o", text)
    text = re.sub("°", "", text)

    # Gestion des symboles
    text = re.sub("[€%$£]", "", text)

    # Gestions des retours à la ligne ou fin de lignes
    text = re.sub('\r\n', " ", text)
    text = re.sub('\n', " ", text)

    # Gestion des espaces
    text = re.sub('\s+', " ", text) # espaces en trop
    text = text.rstrip(" ") # à droite
    text = text.lstrip(" ") # à gauche

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
        pbar.update(1) # actualise la progress bar
    return (texts_out)
    pbar.close()

def tokenize(text):
    text_split = [word for word in text.split()]
    return text_split

stopwords = spacy.lang.en.stop_words.STOP_WORDS

stopwords = [word.lower() for word in stopwords]

# conservation de certains stopwords
liste = [
    # mots à conserver (à exclure de la liste des stopwords par défaut)
]

stopwords = [word for word in stopwords if word not in liste]


def remove_stopwords(text):
    text = [word for word in text if (len(word) > 2) and (word not in stopwords)]
    return text

def detokenize_text(txt):
    txt = ' '.join([word for word in txt])
    return txt

def make_bigrams(data_words_train):
    bigram = gensim.models.Phrases(data_words_train, min_count=5, threshold=0,
                              #  connector_words=stopwords,
                               scoring = 'npmi')
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in data_words_train]

def make_trigrams(data_words_train):
    trigram = gensim.models.Phrases(bigram[data_words_train], threshold=0,
                              #  connector_words=stopwords,
                                scoring = 'npmi')
    bigram = gensim.models.Phrases(data_words_train, min_count=5, threshold=0,
                              #  connector_words=stopwords,
                               scoring = 'npmi')
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in data_words_train]

def transformation(train, trainning ):
    train['location'] = train['location'].astype(str)
    train['city'] = train['location'].apply(city)
    train['country'] = train['location'].apply(country)
    train['precision'] = train['location'].apply(precision)
    train['hashtags'] = train['text'].apply(recupHashtag)
    train['hashtags_b'] = train['text'].apply(recupHashtagBinaire)
    train['names'] = train['text'].apply(recupName)
    train['names_b'] = train['text'].apply(recupNameBinaire)
    train['dates'] = train['text'].apply(recupDate)
    train['dates_b'] = train['text'].apply(recupDateBinaire)
    train['locations'] = train['text'].apply(getLocation)
    train['locations_b'] = train['text'].apply(getLocationBinaire)
    train['rt_path'] = train['text'].apply(getChemin)
    train['rt_path_b'] = train['text'].apply(getCheminBinaire)
    train['text_CLEAN'] = train['text'].apply(lambda x: preprocessing(x))
    train['text_CLEAN_LMT'] = Lemmatization(train,train['text_CLEAN'])
    train['text_CLEAN_LMT_TOKEN'] = train['text_CLEAN_LMT'].apply(lambda x: tokenize(x))
    train['text_CLEAN_LMT_TOKEN_WSW'] = train['text_CLEAN_LMT_TOKEN'].apply(lambda x: remove_stopwords(x))
    data_words_train = train['text_CLEAN_LMT_TOKEN_WSW'].values.tolist()
    train['text_CLEAN_LMT_TOKEN_WSW_BIGRAMS'] = make_bigrams(data_words_train)
    train['text_CLEAN_LMT_WSW_BIGRAMS'] = train['text_CLEAN_LMT_TOKEN_WSW_BIGRAMS'].apply(lambda x: detokenize_text(x))
    vectorizer = TfidfVectorizer()
    if(trainning == True):
        vectorizer_fitted = vectorizer.fit(train['text_CLEAN_LMT_WSW_BIGRAMS'].astype('U'))
        joblib.dump(vectorizer_fitted, 'vectors.gz')
        vectors = vectorizer_fitted.transform(train['text_CLEAN_LMT_WSW_BIGRAMS'].astype('U'))
    else:
        vectorizer_fitted = joblib.load('vectors.gz')
        vectors = vectorizer_fitted.transform(train['text_CLEAN_LMT_WSW_BIGRAMS'].astype('U'))
    feature_names = vectorizer_fitted.get_feature_names_out()
    dense = vectors.todense()
    train_Tfidf = pd.DataFrame(dense, columns=feature_names)
    mean_Tfidf = []
    for feature in train_Tfidf.columns.values:
        mean_Tfidf.append(np.mean(train_Tfidf[feature]))
    train_mean_Tfidf = pd.DataFrame({'word': train_Tfidf.columns.values,'mean_Tfidf': mean_Tfidf})
    threshold = np.percentile(train_mean_Tfidf['mean_Tfidf'],50)
    features_to_delete = train_mean_Tfidf['word'].loc[train_mean_Tfidf['mean_Tfidf'] < threshold]
    for feature in features_to_delete:
        stopwords.extend(feature)
    train['text_CLEAN_LMT_WSW_BIGRAMS_FILTERED'] = [text.split() for text in train['text_CLEAN_LMT_WSW_BIGRAMS'].astype(str)]
    train['text_CLEAN_LMT_WSW_BIGRAMS_FILTERED'] = train['text_CLEAN_LMT_WSW_BIGRAMS_FILTERED'].apply(remove_stopwords)
    train['text_CLEAN_LMT_WSW_BIGRAMS_FILTERED'] = train['text_CLEAN_LMT_WSW_BIGRAMS_FILTERED'].apply(lambda x: detokenize_text(x))
    train_word_features = vectorizer_fitted.fit_transform(train['text_CLEAN_LMT_WSW_BIGRAMS_FILTERED'])
    feature_names = vectorizer_fitted.get_feature_names_out()
    dense = train_word_features.todense()
    train_word_features = pd.DataFrame(dense, columns=feature_names)
    X_train = pd.concat([train_word_features, train[['hashtags_b','names_b','rt_path_b','locations_b']]], axis='columns').reset_index(drop=True)
    if(trainning == True):
        y_train = train['target']
        return X_train, y_train
        print('Préparation entraînement OK')
    else:
        return X_train
        print('Préparation soumission OK')