import pytest
from langdetect import detect

def test_string_text(text):
    rep = '0'
    reg = r"[a-z]|[A-Z]"
    p = re.compile(reg)
    check = re.search(reg,text)
    assert check is not None, "enter a string"

    
def test_letter_repeat(text):
    rep = '0'
    reg = r"(.)\1{4,}"
    p = re.compile(reg)
    check = re.search(reg,text)
    assert check is not None, "you have a repeat letter in your text"


def test_word_repeat(text):
    rep = '0'
    reg = r"\b(\w+)\s+\1\b"
    p = re.compile(reg)
    check = re.search(reg,text)
    assert check is not None, "you have a repeat word in your text"

def test_string_length(text):
    assert len(text) > 10,  "your string doesn't have enough caracter you need to enter "+  str(10-len(text))+ " more character"
    assert len(text) <= 140,  "your string have too many caracter you need to enter "+  str(len(text)-140)+ " less character"

def test_string_english(text):
    assert detect(text) == "en", "you need to entrer an english text"