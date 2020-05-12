from django.shortcuts import render
from .preprocessing.preprocessing import process_data
import pandas as pd
import numpy as np
import pickle
from .models import models
from django.templatetags.static import static
from django.urls import reverse
import xgboost as xgb
import os
import keras
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Index view, returned by the root IP or URL with basic projection description/info and form for fake news analysis
# TODO: Make index contain the intro to the project report, input to accept text, title, and model type info, send to predict through form submission.
def index(request):
    return render(request, 'FND/index.html', {"url": reverse('prediction')})

# Prediction view, returned by form submission from index to present the user with data and selected model prediction.
# TODO: Make prediction accept form from index, process data through pipeline, get prediction from relevant model, give info
def prediction(request):
    model_name_dict = {
        "model1": "TFIDF-Trained XGBoost",
        "model2": "Numerically-Trained XGBoost",
        "model3": None,
        "model4": None,
        "model5": None,
        "model6": None,
    }

    ed_features = ['title_num_letters', 'title_num_words', 'title_perc_up_letters', 'title_perc_up_words',
                    'title_perc_nums', 'title_polarity', 'title_subjectivity', 'text_char_count', 'text_word_count',
                    'text_perc_misspells', 'text_polarity', 'text_subjectivity']

    # Turn form data into a dataframe
    data = request.POST.copy()
    title = data.get('title')
    text = data.get('mytext')
    df = pd.DataFrame([[title, text]], columns=['title', 'text'])
    df_concat = pd.DataFrame([[title+text]], columns=['concat'])

    # preprocess data
    df_proc = process_data(df)

    # Load relevant model and run predictions
    model_name = data.get('mymodel-select')
    model = None
    proba = None
    if model_name == "model1":
        model = models.TfIdfClassifier()
        model_file = '/FND/static/models/tfidf_xgb_fitted.model'
        pkl_file = '/FND/static/models/tfidf_xgb_fitted.pkl'
        model.fit_from_file(model_file, pkl_file)
        proba = int(100*model.predict(df_proc))
    if model_name == "model2":
        model = xgb.Booster({'nthread': 4})
        cwd = os.getcwd()
        model.load_model(cwd + '/FND/static/models/numerical_xgb_eduard.model')
        proba = int(100 * model.predict(xgb.DMatrix(df_proc[ed_features].values))[0])
    if model_name == "model3":
        cwd = os.getcwd()
        tf1 = pickle.load(open(cwd + '/FND/static/models/feature.pkl', 'rb'))
        tfidf = TfidfVectorizer(vocabulary=tf1.vocabulary_)
        X = tfidf.fit_transform(df_concat)
        model = keras.models.load_model(cwd + '/FND/static/models/DNN_please.h5')
        proba = int(100*model.predict(X))

    context = {"proba": proba,
               "model": model_name_dict[model_name],
               }

    # Return prediction
    return render(request, 'FND/prediction.html', context)