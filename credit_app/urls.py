"""
URLs for Credit App
"""
from django.urls import path
from . import views

app_name = 'credit_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.history, name='history'),
    path('about/', views.about, name='about'),
]
