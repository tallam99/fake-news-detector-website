from django.shortcuts import render
from django.http import HttpResponse

# Index View (returned by the root IP or URL)
def index(request):
    return HttpResponse("Welcome to our 460J Project Website!")