import nltk
# nltk.download('nps_chat')
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
    arr=re.split('[?.!]',para)
    # print(arr)
    newpara=[]
    for line in arr:
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
