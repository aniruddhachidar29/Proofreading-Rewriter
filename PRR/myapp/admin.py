from django.contrib import admin
from .models import Sentence, Word, CorWord
# Register your models here.

admin.site.register(Sentence)
admin.site.register(Word)
admin.site.register(CorWord)