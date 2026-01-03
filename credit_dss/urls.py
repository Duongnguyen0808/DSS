"""
URL Configuration for Credit DSS project
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('credit_app.urls')),
]
