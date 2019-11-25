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
import pickle
import urllib
import requests
import json
from pattern.en import conjugate, lemma
from nltk import word_tokenize, pos_tag
import threading

word = "run"
synonym = "enemy"

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
udic={}
def final_spell(numbered_sentences):
    # numbered_sentences=i_s(para)
    spell_dict={}
    l=len(numbered_sentences)

    for i in range(l):
        s=len(numbered_sentences[i])
        if(i+1==l and numbered_sentences[i][s-1][1]!=''):
            continue
        else:
            spell_correction(numbered_sentences[i],spell_dict)
    global udic
    udic=spell_dict
    return spell_dict

stop_words = set(stopwords.words('english'))
def spell_correction(numbered_sentence,spell_dict):

    for word,index in numbered_sentence:
        if(word in stop_words):
            spell_dict[index]=0
        else:
            spell_dict[word]=set(bktree.query(word,1))

    return spell_dict

# dbfile = open('examplePickle', 'rb')
# trigram_freq_dict = pickle.load(dbfile)


iitb_lingo_words=['machau','craxx','infi','scope','lingo','ditch','pain','tum-tum','lukkha','enthu','haga','mugging','farra','ghati','junta','freshie','sophie']
iitb_lingo_meanings=['rocking','cracked','infinite','scopeless','language','ditch','problem','bus','free','enthusiasm','blundered','studying','FR','local_resident','public','freshmen','sophomore']
verb_tag_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
form_verb_list = ["inf", "1sg", "2sg", "3sg", "pl", "part","p", "1sgp", "2sgp", "3gp", "ppl", "ppart"]
pos_tag_lists = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN', 'NNS', 'NNP', 'NNPS']
pseudo_verb_list = ['be', 'do', 'have', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
iitb_lingo_dictionary={}
for i in range(len(iitb_lingo_words)):
    iitb_lingo_dictionary[iitb_lingo_words[i]]=iitb_lingo_meanings[i]

def similarity(a,b):
    wordFromList1 = wordnet.synsets(a)
    wordFromList2 = wordnet.synsets(b)
    if wordFromList1 and wordFromList2:
        s = wordFromList1[0].wup_similarity(wordFromList2[0])

    return s

trigram_freq = {}
final_dict = {}

def synn(word):
    synonyms = []
    for syn in wordnet.synsets(word):
    	for l in syn.lemmas():
            ans=l.name()
            synonyms.append(ans)

    return synonyms

    pass

def synonyms(word):
	verb_form = None
	if verb(word):
		verb_form = verb_ing_ed(word)
		word = lemma(word)
	synons = set([l.name() for syn in wordnet.synsets(word) for l in syn.lemmas()]) # if not l.antonyms()
	if not verb_form:
		return synons
	res = []
	for syn in synons:
		new_syn = conjugate(syn, verb_form)
		res.append(new_syn)
	return set(res)

def synonymss(word):
	verb_form = None
	if verb(word):
		verb_form = verb_ing_ed(word)
		word = lemma(word)
	synonym = synn(word)
	if not verb_form:
		return synonym
	res = []
	for syn in synonym:
		new_syn = conjugate(syn, verb_form)
		res.append(new_syn)
	return set(res)

def verb(word):
	global verb_tag_list
	word_tag = pos_tag_list([word])[0][1]
	if word_tag in verb_tag_list:
		return True
	return False

def verb_ing_ed(word):
	global form_verb_list
	base = lemma(word)
	for alias in form_verb_list:
		if conjugate(base, alias) == word:
			return alias

def pos_tag_list(list_of_words):
	tagged_words = pos_tag(list_of_words)
	return tagged_words

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

def final_synonyms(broke_para):
	global final_dict
	global trigram_freq
	output = {}
    #inefficient taking 60 sec
	# for i in range(len(broke_para)):
	# 	sentence_syms(broke_para[i],output)
	list_of_threads_sentences = []
    #threading for efficiency
	for i in range(len(broke_para)):
		process = threading.Thread(target=sentence_syms, args=(broke_para[i], output))
		process.setDaemon(True)
		process.start()
		list_of_threads_sentences.append(process)
	for process in list_of_threads_sentences:
		process.join()
	list_of_threads_sentences = []
	trigram_freq = {}
	final_dict = {}
	return output

def valid(word):
    if (word.lower() == 'i'):
    	return False
    tag = pos_tag_list([word])[0][1]
    global pos_tag_lists
    global pseudo_verb_list
    if tag in pos_tag_lists:
    	return True
    if verb(word):
    	word = lemma(word)
    	if word in pseudo_verb_list:
    		return False
    	else:
    		return True
    return False

def sentence_syms(sentence, output):
	global final_dict
	words = [word for word, i in sentence]
	list_of_threads = []
	for i in range(len(sentence)):
		if (words[i] == ""):
			continue
		else:
			process = threading.Thread(target=context_syn, args=(words, i, sentence[i][1]))
			process.setDaemon(True)
			process.start()
			list_of_threads.append(process)
			#context_syn(words,i,sentence[i][1])     inefficient
	for process in list_of_threads:
		process.join()
	# print("sen")
	for i in range(len(sentence)):
		if sentence[i][1] in final_dict:
			if final_dict[sentence[i][1]]:
				output[sentence[i][1]] = final_dict[sentence[i][1]]
	pass

def context_syn(words, i, global_key):
	# print('wordbegin')

	global final_dict
	global trigram_freq
	if (words[i] in iitb_lingo_dictionary):
		final_dict[global_key] = [iitb_lingo_dictionary[words[i]]]
		return final_dict[global_key]

	trigramss = trigrams(words, i)
	tri_copy = trigramss
	synonym_list = udic[words[i]]
	score = {}
	threads_list=[]

	for tg, look_word in tri_copy:
		for candidate in synonym_list:
			new_tri = tg[:]
			new_tri[look_word] = candidate
			# print(new_tri)
			api_search = ' '.join(new_tri)
			process = threading.Thread(target=get_freq, args=(api_search, ))
			process.setDaemon(True)
			process.start()
			threads_list.append(process)
			#get_freq(api_search)
			# print('triend')
	# print('end1')
	for process in threads_list:
		process.join()
	# print('end2')
	for tri, target in tri_copy:
		freq = {}
		for candidate in synonym_list:
			new_tri = tri[:]
			new_tri[target] = candidate
			api_search = ' '.join(new_tri)
			# if api_search in trigram_freq_dict:
			# 	freq[candidate] = get_freq(api_search)
			# else:
			# 	freq[candidate]=0
			freq[candidate] = trigram_freq[api_search]
		total_sum = sum(freq[key] for key in freq)
		if total_sum == 0:
			continue
		for key in freq:
			freq[key] /= total_sum
			if key in score:
				score[key] += freq[key]
			else:
				score[key] = freq[key]
	result = sorted(score, key = lambda x: score[x], reverse = True)
	final_dict[global_key] = [ii.lower() for ii in filter(lambda x: (score[x] != 0) and x.lower() != words[i], result)]
	# print('word')
	return final_dict[global_key]

def get_freq(trigram):
    encoded_query = urllib.parse.quote(trigram)
    params = {'corpus': 'eng-us', 'query': encoded_query, 'topk': 3}
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items())

    response = requests.get('https://api.phrasefinder.io/search?' + params)

    assert response.status_code == 200

    response=response.json()

    f = 0
    trigram_freq[trigram] = f
    if response:
    	rest_json = response['phrases']
    	for i in rest_json:
    		f += i['mc']
    		trigram_freq[trigram] = f
    return f

def trigrams(words, i):
	tri = []
	a = words[i - 2 : i + 1]
	if (len(a) == 3):
		tri.append([a, 2])
	a = words[i - 1 : i + 2]
	if (len(a) == 3):
		tri.append([a, 1])
	a = words[i : i + 3]
	if (len(a) == 3):
		tri.append([a, 0])

	return tri

# define training dat
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# 			['this', 'is', 'the', 'second', 'sentence'],
# 			['yet', 'another', 'sentence'],
# 			['one', 'more', 'sentence'],
# 			['and', 'the', 'final', 'sentence']]
# # train model
# model = Word2Vec(sentences, min_count=1)
# # fit a 2d PCA model to the vectors
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
#
#
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# dic={}
# for i, word in enumerate(words):
#     dic[word]=i
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
#
# def similar(a,b):
#     numpy.linalg.norm(result[dic[i],2],resut[dic[i],1])


# while(True):
#     a=input()
#     breaked = word_tokenize(a)
