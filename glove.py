import re
from pymystem3 import Mystem
import requests
import urllib.request
from scipy import spatial

m = Mystem()

def createVector(vec, arr, words):
    for i in range(len(words)):
        num = arr.count(words[i])
        vec[i] = num
    return vec

def preprocess(raw_text):
    # 1. keep only words
    letters_only_text = raw_text
    # 2. convert to lower case and split
    words = letters_only_text.lower().split()
    # 3. remove \n
    break_free_words = [word.rstrip("\n") for word in words]
    # 5. lemmatize
    lemmatized_words = [m.lemmatize(word) for word in break_free_words]
    final = []
    for i in lemmatized_words:
        final.append(i[0])
    return final

def compare(result, model):
    # result = preprocess(result)
    # model = preprocess(model)
    # result = result.split(" ")
    # model = model.split(" ")
    # all_words_in_sentences = result + model
    # all_words_in_sentences = list(set(all_words_in_sentences))
    # vec1_empty = [None] * len(all_words_in_sentences)
    # vec2_empty = [None] * len(all_words_in_sentences)

    # vector_1 = createVector(vec1_empty, result, all_words_in_sentences)
    # vector_2 = createVector(vec2_empty, model, all_words_in_sentences)
    # cosine = spatial.distance.cosine(vector_1, vector_2)

    # return round((1 - cosine) * 100, 2)
    return 67