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

# dbfile = open('examplePickle', 'rb')
# trigram_freq_dict = pickle.load(dbfile)


iitb_lingo_words=['machau','craxx','infi','scope','lingo','ditch','pain','tum-tum','lukkha','enthu','haga','mugging','farra','ghati','junta','freshie','sophie']
iitb_lingo_meanings=['rocking','cracked','infinite','scopeless','language','ditch','problem','bus','free','enthusiasm','blundered','studying','FR','local_resident','public','freshmen','sophomore']
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
word_sugg = {}

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
	if is_verb(word):
		verb_form = conjugated_alias(word)
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
	if is_verb(word):
		verb_form = conjugated_alias(word)
		word = lemma(word)
	synonym = synn(word)
	if not verb_form:
		return synonym
	res = []
	for syn in synonym:
		new_syn = conjugate(syn, verb_form)
		res.append(new_syn)
	return set(res)

def is_verb(word):
	tag_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	word_tag = tag_words([word])[0][1]
	if word_tag in tag_list:
		return True
	return False

def conjugated_alias(word):
	possible_aliases = ["inf", "1sg", "2sg", "3sg", "pl", "part","p", "1sgp", "2sgp", "3gp", "ppl", "ppart"]
	base = lemma(word)
	for alias in possible_aliases:
		if conjugate(base, alias) == word:
			return alias

def tag_words(list_of_words):
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
	global word_sugg
	global trigram_freq
	output = {}
    #inefficient taking 60 sec
	# for i in range(len(broke_para)):
	# 	sentence_syms(broke_para[i],output)
	processes_sentence = []
    #threading for efficiency
	for i in range(len(broke_para)):
		process = threading.Thread(target=sentence_syms, args=(broke_para[i], output))
		process.setDaemon(True)
		process.start()
		processes_sentence.append(process)
	for process in processes_sentence:
		process.join()
	processes_sentence = []
	trigram_freq = {}
	word_sugg = {}
	return output

def valid(word):
    if (word.lower() == 'i'):
    	return False
    tag = tag_words([word])[0][1]
    tag_list = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN', 'NNS', 'NNP', 'NNPS']
    if tag in tag_list:
    	return True
    if is_verb(word):
    	word = lemma(word)
    	lex_list = ['be', 'do', 'have', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
    	if word in lex_list:
    		return False
    	else:
    		return True
    return False

def sentence_syms(sentence, output):
	global word_sugg
	words = [word for word, i in sentence]
	processes_word = []
	for i in range(len(sentence)):
		if (words[i] == "" or not valid(words[i])):
			continue
		else:
			process = threading.Thread(target=context_syn, args=(words, i, sentence[i][1]))
			process.setDaemon(True)
			process.start()
			processes_word.append(process)
			#context_syn(words,i,sentence[i][1])     inefficient
	for process in processes_word:
		process.join()
	# print("sen")
	for i in range(len(sentence)):
		if sentence[i][1] in word_sugg:
			if word_sugg[sentence[i][1]]:
				output[sentence[i][1]] = word_sugg[sentence[i][1]]
	pass

def context_syn(words, i, global_key):
	# print('wordbegin')

	global word_sugg
	global trigram_freq
	if (words[i] in iitb_lingo_dictionary):
		word_sugg[global_key] = [iitb_lingo_dictionary[words[i]]]
		return word_sugg[global_key]

	trigramss = trigrams(words, i)
	filtered_trigrams = trigramss
	synonym_list = synonyms(words[i])
	score = {}
	processes=[]

	for tg, look_word in filtered_trigrams:
		for candidate in synonym_list:
			new_tri = tg[:]
			new_tri[look_word] = candidate
			# print(new_tri)
			string_tri = ' '.join(new_tri)
			process = threading.Thread(target=get_freq, args=(string_tri, ))
			process.setDaemon(True)
			process.start()
			processes.append(process)
			#get_freq(string_tri)
			# print('triend')
	# print('end1')
	for process in processes:
		process.join()
	# print('end2')
	for tri, target in filtered_trigrams:
		freq = {}
		for candidate in synonym_list:
			new_tri = tri[:]
			new_tri[target] = candidate
			string_tri = ' '.join(new_tri)
			# if string_tri in trigram_freq_dict:
			# 	freq[candidate] = get_freq(string_tri)
			# else:
			# 	freq[candidate]=0
			freq[candidate] = trigram_freq[string_tri]
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
	word_sugg[global_key] = [ii.lower() for ii in filter(lambda x: (score[x] != 0) and x.lower() != words[i], result)]
	# print('word')
	return word_sugg[global_key]

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
	res = []
	a = words[i - 2 : i + 1]
	if (len(a) == 3):
		res.append([a, 2])
	a = words[i - 1 : i + 2]
	if (len(a) == 3):
		res.append([a, 1])
	a = words[i : i + 3]
	if (len(a) == 3):
		res.append([a, 0])

	return res

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
