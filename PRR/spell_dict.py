import nltk
import time
# from nltk.corpus import words
# from itertools import imap, ifilter

from nltk.tokenize import sent_tokenize, word_tokenize
mistake = "zebr hmework what waiter"
breaked = word_tokenize(mistake)
words1 = nltk.corpus.words.words()

words1.sort();

def levenshtein2(s, t):
    m, n = len(s), len(t)
    d = [range(n+1)]
    d += [[i] for i in range(1,m+1)]
    for i in range(0,m):
        for j in range(0,n):
            cost = 1
            if s[i] == t[j]: cost = 0

            d[i+1].append( min(d[i][j+1]+1, # deletion
                               d[i+1][j]+1, #insertion
                               d[i][j]+cost) #substitution
                           )
    return d[m][n]


final=[]
for i in range(len(words1)):
	arr=[]
	for j in range(len(words1)):
		arr.append(levenshtein2(words1[i],words1[j]))
	final.append(arr)

