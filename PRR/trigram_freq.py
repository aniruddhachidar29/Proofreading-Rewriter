import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy
import string
import urllib.request
import re
import csv

# f = open("w3_.txt", "r")

dictionary={}

# for x in f:
freq=[]
with open ('w3c.txt', 'r') as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        #arr.append(row)
        str=" ".join(row[1:4])
        dictionary[str.lower()]=row[0]

import urllib
import requests

encoded_query = urllib.parse.quote('how ?')
params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 3}
params = '&'.join('{}={}'.format(name, value) for name, value in params.items())

response = requests.get('https://api.phrasefinder.io/search?' + params)

assert response.status_code == 200

print(response.json())
