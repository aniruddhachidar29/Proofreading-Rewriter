from django.shortcuts import render
from .models import Sentence, Word, CorWord
#from '../../Spell_Checker.py' import
#from django.http import Http404

# Create your views here.
def index(request):
	#Sentence.objects.all().delete()
	if request.method == 'POST':
		Sentence.objects.all().delete()
		sent = Sentence(sentence = request.POST['name_text'])
		print(request.POST['name_text'])
		sent.save()
	all_sentences = Sentence.objects.all()
	all_words = Word.objects.all()
	all_corWords = CorWord.objects.all()
	context = {'sent': all_sentences, 'wor': all_words, 'cor': all_corWords}
	return render(request, "myapp/index.html", context)