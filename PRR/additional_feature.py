import nltk
#nltk.download('nps_chat')
from nltk.corpus import brown
from nltk.tokenize.treebank import TreebankWordDetokenizer

brown_tagged_sents=brown.tagged_sents()
brown_sents=brown.sents()
ut=nltk.UnigramTagger(brown_tagged_sents)

posts = nltk.corpus.nps_chat.xml_posts()[:10000]
posts2 = nltk.corpus.nps_chat.xml_posts()[:]
import re

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts2]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(classifier, test_set))
def is_question(line):
    a=classifier.classify(dialogue_act_features(line))
    question=False
    if(a=='ynQuestion' or a=='whQuestion'):
        question=True
    # print(a)
    return question

def punctuation(para):
    fin = []
    #words = nltk.tokenize.word_tokenize(para)
    words = re.split('[ ]',para)
    cap_sugg = {}
    for ind in range(len(words)) :
        word = words[ind]
        flag = 0
        ch = ''
        if (word == ""):
            continue
        if (word[-1]=='.' or word[-1]=='?' or word[-1]=='!'):
            flag = 1
            ch = word[-1]
            word=word[:-1]
        sug = word
        if (ut.tag([word])[0][1] is None):
            c=word[0]
            word1=word[0].upper()+word[1:]
            cap_sugg[ind]=word1
            sug = word1
        sug=sug+ch
        fin.append(sug)
    para = TreebankWordDetokenizer().detokenize(fin)
    print(para)
    arr=re.split('[?.!]',para)
    # print(arr)
    newpara=[]
    for line in arr:
        line=line.lstrip()
        line=line.rstrip()
        if(line==''):
            continue
        if(is_question(line)):
            newline=line[0].upper()+line[1:len(line)]
            newpara.append(newline+'?')
        else:
            newline=line[0].upper()+line[1:len(line)]
            newpara.append(newline+'.')
    if(newpara[len(newpara)-1]=='.'):
        newpara=newpara[:len(newpara)-1]
    str=' '.join(newpara)
    # print(str)
    # print(str)
    return str
