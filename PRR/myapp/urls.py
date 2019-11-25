from django.urls import path, re_path
from . import views

urlpatterns = [
	
	#/music/
	path('', views.index, name ='index'),

	#/music/712/
	#re_path(r'^app/', 'myapp.views.app', name = 'app')
]