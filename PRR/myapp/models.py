from django.db import models

# Create your models here.
class Sentence(models.Model):
	sentence = models.CharField(max_length = 5000)
	complete = models.BooleanField(default = False)
	changed = models.BooleanField(default = False)
	#endsym = models.CharField(max_length = 1)#?,!.

	def __str__(self):
		return self.sentence

class Word(models.Model):
	sentence = models.ForeignKey(Sentence, on_delete = models.CASCADE)
	word = models.CharField(max_length = 50)
	sym = models.BooleanField(default = False)
	
	def __str__(self):
		return self.word

class CorWord(models.Model):
	word = models.ForeignKey(Word, on_delete = models.CASCADE)
	corWord = models.CharField(max_length = 50)

	def __str__(self):
		return self.corWord