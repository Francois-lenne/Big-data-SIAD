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
import nltk
from nltk.stem import PorterStemmer
nlp = spacy.load("en_core_web_sm")
stopwords = spacy.lang.en.stop_words.STOP_WORDS

def clean(text):
   
    pattern = re.compile('[^a-zA-Z]')
    words = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [PorterStemmer().stem(word) for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text



def DataPreparation(train, sample_type = 'train'):

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

    # Détecteur de vulgarités
    def isProfanity(text): 
      from better_profanity import profanity 
      check = profanity.contains_profanity(text)
      if check == "false":
          profanity = 0
      else:
          profanity = 1
      return(profanity)

    # Récupération des hashtag
    def recupHashtag(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant une chaîne contenant les hashtags séparés par un espace (s'ils existent).
        """
        reg = r"#(\w+)"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
            ge = p.findall(text)
            ge_join = ' '.join(ge)
            return ge_join

    # Variable binaire : présence ou non d'un hashtag
    def recupHashtagBinaire(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant "0" si aucun hashtag n'est présent et "1" si un hashtag est présent.
        """
        rep = '0'
        reg = r"#(\w+)"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
          rep = '1'
        return rep

    # Décompte des hashtags
    def countHashtag(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant retournant le nombre de hashtag # dans le tweet.
        """
        rep = 0
        reg = r"#(\w+)"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
          rep = len(re.search(reg,text).groups())
        return rep

    def recupNameBinaire(text):
        rep = '0'
        reg = r"@(\w+)"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
            rep = '1'
        return rep

    # Récupération des mentions (@)
    def recupName(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant une chaîne contenant les mentions séparées par un espace (si elles existent).
        """
        reg = r"@(\w+)"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
            ge = p.findall(text)
            ge_join = ' '.join(ge)
            return ge_join

    # Variable binaire : présence d'une mention (@)
    def countName(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant le nombre de mentions @ dans le tweet.
        """
        rep = 0
        reg = r"@(\w+)"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
          rep = len(re.search(reg,text).groups())
        return rep

    # Récupération des dates
    def recupDate(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant une chaîne contenant une date (si elle existe).
        """
        reg = r"([A-Za-z]{3})\s(\d{1,2}),\s(\d{4})"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
            ge = p.findall(text)
            return ge

    # Variable binaire : présence d'une date
    def recupDateBinaire(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant "0" si aucune date n'est présente et "1" si une date est présente.
        """
        rep = '0'
        reg = r"([A-Za-z]{3})\s(\d{1,2}),\s(\d{4})"
        p = re.compile(reg)
        check = re.search(reg,text)
        if check is not None:
          rep = '1'
        return rep

    # Récupération des liens
    def getChemin(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant une chaîne contenant le chemin du lien (s'il existe).
        """
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

    # Variable binaire : présence ou non d'un lien
    def getCheminBinaire(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant "0" si aucun hashtag n'est présent et "1" si un hashtag est présent.
        """
        rep = '0'
        reg1 = r'(https?:\/\/[^\s]+)'
        p = re.compile(reg1)
        check = re.search(reg1,text)
        if check is not None:
          rep = '1'
        return rep

    # Variable binaire : présence ou non d'un lien
    def countChemin(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant "0" si aucun hashtag n'est présent et "1" si un hashtag est présent.
        """
        rep = 0
        reg1 = r'(https?:\/\/[^\s]+)'
        p = re.compile(reg1)
        check = re.search(reg1,text)
        if check is not None:
          rep = len(re.search(reg1,text).groups())
        return rep

    # Récupération des lieux cités
    def getLocation(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant une liste de lieu(x) (s'il(s) existe(nt)).
        """
        global nlp
        save = []
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                save.append(ent.text)
        return save

    # Variable binaire : présence ou non d'un lieu
    def getLocationBinaire(text):
        """
        Fonction prenant en entrée une chaîne de caractère et retournant "0" si aucun hashtag n'est présent et "1" si un hashtag est présent.
        """
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

    def removeElements(text):
      text = re.sub('([A-Za-z]{3})\s(\d{1,2}),\s(\d{4})','',text)
      text = re.sub('@(\w+)','',text)
      text = re.sub('#(\w+)','',text)
      text = re.sub('(https?:\/\/[^\s]+)','',text)
      text = re.sub('(https?)://([^:/]+)(?::(\d+))?(/[^?]*)?(\?[^#]*)?(#.*)?','',text)

    def countEmojis(text):
        rep = 0
        reg = r"#(\w+)"
        p = re.compile(        '['
                u'\U0001F600-\U0001F64F'  # emoticons
                u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                u'\U0001F680-\U0001F6FF'  # transport & map symbols
                u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                u'\U00002702-\U000027B0'
                u'\U000024C2-\U0001F251'
                ']+',
                flags=re.UNICODE)
        check = re.search(reg,text)
        if check is not None:
          rep = len(re.search(reg,text).groups())
        return rep

    def preprocessing(text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'&amp|& amp', '&', text) # remplacement &amp par &
        # text = re.sub("\d+", " ", text) # retrait nombres
        text = re.sub('[éèê]', "e", text) # retrait accents sur le e
        text = re.sub("ôöóò", "o", text) # retrait accents sur le o
        text = re.sub("üùû", "u", text) # retrait accents sur le u
        text = re.sub("ïiî", "i", text) # retrait accents sur le i
        text = re.sub("âàäå", "a", text) # retrait accents sur le a
        text = re.sub("[.,;:!?]", " ", text) # retrait ponctuation
        text = re.sub("[|{}()«»/]", " ", text) # retrait parenthèses, guillemets, slashs...
        text = re.sub("[“”]", " ", text) # retrait guillemets (autre forme)
        text = re.sub("'", " ", text) # retrait apostrophes
        text = re.sub("’", " ", text) # retrait apostrophes (autre forme)
        text = re.sub('"', " ", text) # retrait quotes
        text = re.sub('[+-]', " ", text) # retrait + et -
        text = re.sub('[=*/]', " ", text) # retrait opérateurs
        text = re.sub("°", "", text) # retrait symbole °
        text = re.sub("[€%$£]", "", text) # retrait symboles devises
        text = re.sub('\r\n', " ", text) # retrait retour charriot/retour à la ligne
        text = re.sub('\n', " ", text) # retrait retour à la ligne
        text = re.sub('\s+', " ", text) # retrait espaces en trop
        text = text.rstrip(" ") # retrait espaces à droite
        text = text.lstrip(" ") # retrait espaces à gauche
        emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
        emoji_pattern.sub(r'', text)
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
        text = [word for word in text if (len(word) > 2) and (word not in stopwords)]
        return text

    train['location'] = train['location'].fillna('')
    train['keyword'] = train['keyword'].fillna('')
    train['nb_emojis'] = train['text'].apply(countEmojis)
    train['nb_hashtags'] = train['text'].apply(countHashtag)
    train['nb_mentions'] = train['text'].apply(countName)
    train['nb_liens'] = train['text'].apply(countChemin)
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
    train['location'] = train['text'].apply(removeElements)
    train['Subjectivity'] = train['text'].apply(getSubjectivity)
    train['Polarity'] = train['text'].apply(getPolarity)
    train['text_CLEAN'] = train['text'].apply(lambda x: preprocessing(x))
    train['keyword'] = train['keyword'].apply(lambda x: preprocessing(x))
    train['text_length'] = [len(text) for text in train['text']]
    train['tweet_cut'] = pd.cut(train['text_length'], [0,100,140], labels=['<100','>100'])
    train['text_CLEAN_LMT'] = Lemmatization(train,train['text_CLEAN'])
    train['text_CLEAN_LMT_TOKEN'] = train['text_CLEAN_LMT'].apply(lambda x: tokenize(x))

    nlp = spacy.load('en_core_web_sm')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
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
    
    # Fitter le vectorizer en amont sur les données d'entraînement
    vectorizer = joblib.load('vectorizer.gz')
    vectors = vectorizer.transform(train['text_CLEAN_LMT_WSW_BIGRAMS'].astype('U'))
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    train_word_features = pd.DataFrame(dense, columns=feature_names)


    train_variables_extracted = train[['hashtags_b','names_b','rt_path_b','locations_b', 'Subjectivity', 'Polarity']]
    train_location = train[['city','country','precision', 'tweet_cut']]

    final_train = pd.concat([train_word_features, # variables créées à l'aide des tweets (features TF IDF)
                             train_location, # variables créées à partir des localisations
                             train_variables_extracted], # variables créées via l'extraction de données dans les tweets
                             axis='columns')

    X_train = final_train.reset_index(drop=True)

    if sample_type == 'train':
        y_train = train['target']
        return X_train, y_train
    
    else:
        return X_train