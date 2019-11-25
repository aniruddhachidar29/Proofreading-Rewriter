from django.shortcuts import render
from .models import Sentence, Word, CorWord
import re
from Spell_Checker import *
from grammar_checker import *
from synonyms import *
import pattern
#from '../../Spell_Checker.py' import
#from django.http import Http404

# Create your views here.
def index(request):
	#Sentence.objects.all().delete()
	if request.method == 'POST':
		Sentence.objects.all().delete()
		para = request.POST['name_text']
		
		parr = []
		for i in para:
			if i =='?' or i == '.' or i == '!' or i == ',':
				parr.append(i)
		#print(parr)	
		list_sent = re.split('[.,?!]',para)
		n = len(list_sent)
		lst = []
		if list_sent[-1] != "":
			lst = list_sent[:n-1]
			last = list_sent[-1]
			for s in lst:
				s1 = Sentence(sentence = s, changed = True, complete = True)
				s1.save()
			s1 = Sentence(sentence = last, changed = True)
		else:
			lst = list_sent[:n-1]
			for s in lst:
				s1 = Sentence(sentence = s, changed = True, complete = True)
				s1.save()
		counter = 0
		for s in lst:
			counter = counter + 1
			word_list = re.split('[ ]',s) 
			#print(word_list)
			s = Sentence.objects.all()[counter-1]
			for wr in word_list:
				if wr != "":
					w = Word(word = wr, sentence = s)
					w.save()
					
			w = Word(word = parr[counter-1], sentence = s, sym = True)
			w.save()

		spelldict = final_spell(para)
		wordpk = 1
		for key, value in spelldict.items():
			w = Word.objects.all()[wordpk -1]
			#print(wordpk)
			#print(value)
			#print(key)
			#print(w.sym)
			if w.sym == True:
				wordpk = wordpk + 1


			if value == []:
				pass
				#print(value)
			else:
				for corword in value:
					cw = CorWord(word = Word.objects.all()[wordpk-1], corWord = corword)
					cw.save()
					#print("hi")
			wordpk = wordpk +1

		gramdict = grammar_check(para)
		wordpk = 1
		for key, value in spelldict.items():
			w = Word.objects.all()[wordpk -1]
			#print(wordpk)
			#print(value)
			#print(key)
			#print(w.sym)
			if w.sym == True:
				wordpk = wordpk + 1


			if value == []:
				pass
				#print(value)
			else:
				for corword in value:
					cw = CorWord(word = Word.objects.all()[wordpk-1], corWord = corword)
					cw.save()
					#print("hi")
			wordpk = wordpk +1

		syndict = final_synonyms(para)
		wordpk = 1
		for key, value in spelldict.items():
			w = Word.objects.all()[wordpk -1]
			#print(wordpk)
			#print(value)
			#print(key)
			#print(w.sym)
			if w.sym == True:
				wordpk = wordpk + 1


			if value == []:
				pass
				#print(value)
			else:
				for corword in value:
					cw = CorWord(word = Word.objects.all()[wordpk-1], corWord = corword)
					cw.save()
					#print("hi")
			wordpk = wordpk +1



	all_sentences = Sentence.objects.all()
	all_words = Word.objects.all()
	all_corWords = CorWord.objects.all()

	
	context = {'sent': all_sentences, 'wor': all_words, 'cor': all_corWords}
	return render(request, "myapp/index.html", context)