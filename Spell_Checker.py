import nltk
import time
# from nltk.corpus import words
# from itertools import imap, ifilter
import re
import pickle
from nltk.corpus import stopwords 
import urllib
import requests
from nltk.tokenize import sent_tokenize, word_tokenize
mistake = "zebr hmework what waiter"
breaked = word_tokenize(mistake)
words1 = nltk.corpus.words.words()

words1.sort();

words2 = ['apple', 'bag', 'drawing', 'listing', 'linking', 'living', 'lighting', 'orange', 'walking', 'zoo']

tolerability=1

arr=[]

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper
def memoize(func):
    mem = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]
    return memoizer
@call_counter
@memoize
def levenshtein(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1

    res = min([levenshtein(s[:-1], t)+1,
               levenshtein(s, t[:-1])+1,
               levenshtein(s[:-1], t[:-1]) + cost])
    return res

global_arr=[]
c=0
with open('bktree', 'rb') as f:
    dist=pickle.load(f)

class BKTree:
    def __init__(self, distfn, words):

        self.distfn = distfn

        it = iter(words)
        # root = it.next()
        root = next(it)
        self.tree = (root, {})

        for i in it:
            self._add_word(self.tree, i)

    def _add_word(self, parent, word):
        pword, children = parent
        # d = self.distfn(word, pword)
        global c
        d=dist[c]
        c=c+1
        # global_arr.append(d)
        if d in children:
            self._add_word(children[d], word)
        else:
            children[d] = (word, {})

    def query(self, word, n):

        def rec(parent):
            pword, children = parent
            if(len(word)-len(word) > n):
                d=n+1
            else:
                d = self.distfn(word, pword)
            results = []
            if d <= n:
                results.append( pword )

            for i in range(d-n, d+n+1):
                child = children.get(i)
                if child is not None:
                    results.extend(rec(child))
            return results

        # sort by distance
        return sorted(rec(self.tree))

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

def editDistDP(str1, str2):
    # Create a table to store results of subproblems
    m=len(str1)
    n=len(str2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)]

    # Fill d[][] in bottom up manner
    for i in range(m+1):
        for j in range(n+1):

            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace

    return dp[m][n]

breaked = word_tokenize(mistake)

def i_s(p):
    sentences = re.split('[.,?!]', p);
    # print(sentences)
    n = 0
    iss = []
    for sentence in sentences:
        sentence = sentence.lstrip()
        words = sentence.split(' ')
        isa = []
        for word in words:
            isa.append((word, n))
            n += 1
        # print(indexed_sentence)
        iss.append(isa)
    return iss
    pass

def get_freq(trigram):
    encoded_query = urllib.parse.quote(trigram)
    params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 3}
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items())

    response = requests.get('https://api.phrasefinder.io/search?' + params)

    assert response.status_code == 200

    response=response.json()

    f = 0
    if response:
        rest_json = response['phrases']
        for i in rest_json:
            f += i['mc']
    return f

bktree = BKTree(levenshtein2,words1)
print('start')

def final_spell(para):
    numbered_sentences=i_s(para)
    spell_dict={}
    l=len(numbered_sentences)
    
    for i in range(l):
        s=len(numbered_sentences[i])
        if(i+1==l and numbered_sentences[i][s-1][1]!=''):
            continue
        else:
            spell_correction(numbered_sentences[i],spell_dict)

    return spell_dict

stop_words = set(stopwords.words('english')) 
def spell_correction(numbered_sentence,spell_dict):
    
    for word,index in numbered_sentence:
        if(word in stop_words):
            spell_dict[index]=0
        else:
            spell_dict[index]=set(bktree.query(word,1))

    return spell_dict

# while(True):
#     a=input()
#     start=time.time()
#     print(bktree.query(a,1))
#     end=time.time()
#     print(end-start)


# for l in words2:
#     print(bktree.query(l,tolerability))

# print(bktree.query("ligting",tolerability))

# if __name__ == "__main__":
#
#     tree = BKTree(levenshtein,
#                   dict_words('/usr/share/dict/american-english-large'))
#
#     print tree.query("ricoshet", 2)

#     dist = 1
#     for i in ["book", "cat", "backlash", "scandal"]:
#         w = set(tree.query(i, dist)) - set([i])
#         print "words within %d of %s: %r" % (dist, i, w)
