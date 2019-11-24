import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import json
import string
from pattern.en import conjugate, lemma
import urllib.request
import threading
import requests
import numpy
import string
import urllib.request
import re
import pickle
import urllib
import requests
import json

final_sugg = {}
def grammar_check(para):
	sentences = i_s(para)
	for sentence in sentences:
		sentence = [(word, index) for word, index in sentence if word != ""]
		sentence_string = [word for word, index in sentence]
		tagged_sentence = pos_tag(sentence_string)
		for indt in range(len(sentence)):
			alternatives = suggs(sentence_string, indt, tagged_sentence[indt][1])
			if alternatives:
				final_sugg[sentence[indt][1]] = alternatives
	return final_sugg

trigram_freq = {}
def get_frq(trigram):
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

def i_s(p):
    sentences = re.split('[.,?!]', p);
    n = 0
    iss = []
    for sentence in sentences:
        sentence = sentence.lstrip()
        words = sentence.split(' ')
        isa = []
        for word in words:
            isa.append((word, n))
            n += 1
        iss.append(isa)
    return iss
    pass

articles_list = ['a', 'an', 'the']
interrogative_words=['why','what','when','which','whose','whom','how','where','that','who']
do_verbs=['do','does','done']
hv_verbs=['has','have','had']
be_verbs=['be','are','is','were','was','been','being','am']
modes = ["inf", "1sg", "2sg", "3sg", "pl", "part","p", "1sgp", "2sgp", "3gp", "ppl", "ppart"]

def alters(word,tag):
	base_form = lemma(word)
	s=tag[0:2]
	if(s=='VB'):
		alternative = [conjugate(base_form, mode) for mode in modes if conjugate(base_form, mode) is not None]
		return alternative
	elif(s=='BE'):
		return be_verbs
	elif(s=='HV'):
		return hv_verbs
	elif(s=="DO"):
		return do_verbs
	elif(tag[0]=="W"):
		return interrogative_words
	elif(s=="DT"):
		return articles_list

processes = []
limit = 1.9
def suggs(list_of_words, indt, tag):
	if (tag not in ['DT', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','HVD','HVG','HVN','HVZ','HV','BE','BER','BEZ','BED','BEDZ','BEG','BEM','BEN','DOD','DO','DOZ','WDT','WP$','WPO','WPS','WQL','WRB']):
		return []
	global processes
	trigrams = adj_trigrams(list_of_words, indt,tag)
	alternative_list = alters(list_of_words[indt],tag)
	score = {}

	for tri, tloc in trigrams:
		for repl in alternative_list:
			new_tri = tri[:]
			new_tri[tloc] = repl
			string_tri = ' '.join(new_tri)
			#r = get_frq(string_tri)
			process = threading.Thread(target=get_frq, args=(string_tri, ))
			process.setDaemon(True)
			process.start()
			processes.append(process)
	for process in processes:
		process.join()
	for tri, tloc in trigrams:
		freq = {}
		for repl in alternative_list:
			new_tri = tri[:]
			new_tri[tloc] = repl
			string_tri = ' '.join(new_tri)
			freq[repl] = trigram_freq[string_tri]
		total_sum = sum(freq[key] for key in freq)
		if total_sum == 0:
			continue
		for key in freq:
			freq[key] /= total_sum
			if key in score:
				score[key] += freq[key]
			else:
				score[key] = freq[key]
	processes = []
	result = sorted(score, key = lambda x: score[x], reverse = True)
	return [i.lower() for i in filter(lambda x: (score[x] != 0) and x.lower() != list_of_words[indt] and score[x] > limit * score[list_of_words[indt].lower()], result)]

def adj_trigrams(list_of_words, indt,tag):
	res = []
	if(tag[0:2]=='VB'):
		pref = list_of_words[max(0, indt - 2) : indt + 2]
		res.append([pref, len(pref) - 2])
	else:
		nbd = list_of_words[max(0, indt - 1) : indt + 2]
		if(indt+2>len(list_of_words)):
			res.append([nbd, len(nbd) - 1])
		else:
			res.append([nbd, len(nbd) - 2])
	return res
