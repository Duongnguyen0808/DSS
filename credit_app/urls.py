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
    path('ahp-matrix/', views.ahp_matrix, name='ahp_matrix'),
    path('ahp-matrix/save/', views.save_custom_matrix, name='save_custom_matrix'),
    path('ahp-matrix/reset/', views.reset_matrix, name='reset_matrix'),
    path('ahp-alternatives/', views.ahp_alternatives, name='ahp_alternatives'),
    path('about/', views.about, name='about'),
]
