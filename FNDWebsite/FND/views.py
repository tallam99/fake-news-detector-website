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
        proba = model.predict(df_proc)
    if model_name == "model2":
        model = xgb.Booster({'nthread': 4})
        cwd = os.getcwd()
        model.load_model(cwd + '/FND/static/models/numerical_xgb_eduard.model')
        proba = int(100 * model.predict(xgb.DMatrix(df_proc[ed_features].values))[0])
    if model_name == "model3":
        pass
    if model_name == "model4":
        pass
    if model_name == "model5":
        pass
    if model_name == "model6":
        pass


    # title_num_letters = df_proc['title_num_letters'].values[0]
    # title_num_words = df_proc['title_num_words'].values[0]
    # title_perc_up_letters = df_proc['perc_up_letters'].values[0]
    # title_perc_up_words = df_proc['title_perc_up_words'].values[0]
    # title_perc_nums = df_proc['title_perc_nums'].values[0]
    # title_polarity = df_proc['title_polarity'].values[0]
    # title_subjectivity = df_proc['title_subjectivity'].values[0]
    # text_char_count = df_proc['text_char_count'].values[0]
    # text_word_count = df_proc['text_word_count'].values[0]
    # text_cap_word_count = df_proc['text_cap_word_count'].values[0]
    # text_perc_misspells = df_proc['text_perc_misspells'].values[0]
    # text_polarity = df_proc['text_polarity'].values[0]
    # text_subjectivity = df_proc['text_subjectivity'].values[0]

    context = {"proba": proba,
               "model": model_name_dict[model_name],
               # "title_num_letters": title_num_letters,
               # "title_num_words": title_num_words,
               # "title_perc_up_letters": title_perc_up_letters,
               # "title_perc_up_words": title_perc_up_words,
               # "title_perc_nums": title_perc_nums,
               # "title_polarity": title_polarity,
               # "title_subjectivity": title_subjectivity,
               # "text_char_count": text_char_count,
               # "text_word_count": text_word_count,
               # "text_cap_word_count": text_cap_word_count,
               # "text_perc_misspells": text_perc_misspells,
               # "text_polarity": text_polarity,
               # "text_subjectivity": text_subjectivity,
               }

    # Return prediction
    return render(request, 'FND/prediction.html', context)