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

word = "run"
synonym = "enemy"

dbfile = open('examplePickle', 'rb')
trigram_freq_dict = pickle.load(dbfile)


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

def syn(word):
    synonyms = []
    for syn in wordnet.synsets(word):
    	for l in syn.lemmas():
            ans=l.name()
            synonyms.append(ans)

    return synonyms

    pass

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
	for i in range(len(broke_para)):
		sentence_syms(broke_para[i],output)
	trigram_freq = {}
	word_sugg = {}
	return output

def invalid(words):
    return False

def sentence_syms(sentence, output):
	global word_sugg
	words = [word for word, i in sentence]
	for i in range(len(sentence)):
		if (words[i] == "" or invalid(words[i])):
			continue
		else:
			context_syn(words,i,sentence[i][1])
	for i in range(len(sentence)):
		if sentence[i][1] in word_sugg:
			if word_sugg[sentence[i][1]]:
				output[sentence[i][1]] = word_sugg[sentence[i][1]]
	pass

def context_syn(words, i, global_key):
	global word_sugg
	global trigram_freq
	if (words[i] in iitb_lingo_dictionary):
		word_sugg[global_key] = [iitb_lingo_dictionary[words[i]]]
		return word_sugg[global_key]

	trigramss = trigrams(words, i)
	filtered_trigrams = trigramss
	synonym_list = syn(words[i])
	score = {}

	# for tg, look_word in filtered_trigrams:
	# 	for candidate in synonym_list:
	# 		new_tri = tg[:]
	# 		new_tri[look_word] = candidate
	# 		string_tri = ' '.join(new_tri)
	# 		get_freq(string_tri)

	for tri, target in filtered_trigrams:
		freq = {}
		for candidate in synonym_list:
			new_tri = tri[:]
			new_tri[target] = candidate
			string_tri = ' '.join(new_tri)
			if string_tri in trigram_freq_dict:
				freq[candidate] = float(trigram_freq_dict[string_tri])
			else:
				freq[candidate]=0
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
	word_sugg[global_key] = [i.lower() for i in filter(lambda x: (score[x] != 0) and x.lower() != words[i], result)]
	return word_sugg[global_key]


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
