import nltk
from nltk.corpus import wordnet



word = "run"
synonym = "enemy"

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

print(set(syn(word)))
