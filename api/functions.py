# Les librairies importées :
# - spacy contient des dictionnaires destinées au NLP(Natural Language Processing)
# - nltk contient des fonctions destinées au NLP(Natural Language Processing)

import spacy
import nltk
from nltk.stem import PorterStemmer
nlp = spacy.load("en_core_web_sm")
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Fonction clean
# Tokenisation du texte, supression des stopwords, remplacement des mots par leur racine, jointure entre les mots.

def clean(text):
    words = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [PorterStemmer().stem(word) for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text




