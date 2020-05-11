from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('prediction/', prediction, name='prediction'),
]