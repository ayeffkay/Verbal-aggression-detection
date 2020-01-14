"""Tokenization, stemming and lemmatization module"""

import re
from mystem import analyze
from pymystem3 import Mystem
import pymorphy2
from nltk.tokenize import regexp_tokenize
from nltk.stem.snowball import RussianStemmer
import preprocessing


def tokenize(corpus):
    filtered = []
    for comment in corpus:
        comment = comment.lower()
        words = regexp_tokenize(comment, pattern=re.compile("[а-яё]+", re.IGNORECASE))
        filtered.append(' '.join(words))
    return filtered


# Snowball stemmer
def stemming(corpus):
    stemmer = RussianStemmer()
    stems = []
    for comment in corpus:
        comment = comment.split()
        s = [stemmer.stem(word) for word in comment]
        stems.append(' '.join(s))
    return stems


def lemmatization1(corpus):
    m = Mystem()
    lemmas = []
    for comment in corpus:
        lem = m.lemmatize(comment)
        res = ''.join(lem).rstrip()
        lemmas.append(''.join(res).rstrip())
    return lemmas


def pymorphy_lem(word):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(word)[0]
    return p.normal_form, p.tag


def mystem_lem(word):
    res = analyze(word)
    with res as r:
        normal_form = r[0].form
        gram = r[0].stem_grammemes
    gram = [str(gr) for gr in gram]
    return normal_form, gram


def in_constraint(constr, tag):
    for c in constr:
        if c in tag:
            return True
    return False


def lemmatization2(corpus, pos1=preprocessing.pos1,
                   pos2=preprocessing.pos2,
                   constr1=preprocessing.constr1,
                   constr2=preprocessing.constr2):
    lemmas = []
    morph = pymorphy2.MorphAnalyzer()
    for text in corpus:
        s = ""
        for word in text.split():
            p = morph.parse(word)[0]
            n2, tag2 = mystem_lem(word)
            if (p.tag.POS in pos1) or (tag2 and tag2[0] in pos2):
                if not (in_constraint(constr1, p.tag)
                        or in_constraint(constr2, tag2)):
                    s += p.normal_form + " "
        lemmas.append(s)
    return lemmas
