from django.shortcuts import render
from django.http import HttpResponse

# Index view, returned by the root IP or URL with basic projection description/info and form for fake news analysis
# TODO: Make index contain the intro to the project report, input to accept text, title, and model type info, send to predict through form submission.
def index(request):
    return HttpResponse("Welcome to our 460J Project Website!")

# Prediction view, returned by form submission from index to present the user with data and selected model prediction.
# TODO: Make prediction accept form from index, process data through pipeline, get prediction from relevant model, give info
def prediction(request):
    return HttpResponse("I can't predict anything yet.")