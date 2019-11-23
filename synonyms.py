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

word = "run"
synonym = "enemy"

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
	processes_sentence = []
	for i in range(len(broke_para)):
        get_synonyms_sentence(broke_para[i],output)
	processes_sentence = []
	trigram_freq = {}
	word_sugg = {}
	return output

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
