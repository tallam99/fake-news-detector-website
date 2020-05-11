from django.urls import path
from FNDWebsite.FND.views import *

urlpatterns = [
    path('', index, name='index'),
]