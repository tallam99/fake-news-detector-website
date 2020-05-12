# Overall Preprocessing
# Fake News Detector EE460J Final Project
# By Jackson Lightfoot

import numpy as np
import pandas as pd
from .cleaning import *
from .title_extraction import *
from .text_extraction import *
from .concat_extraction import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def process_data(df):

    clean_pipe = Pipeline([
        ('title_novideo_noreuters', TitleNoVideoNoReuters()),
        ('title_nopunc', TitleNoPunc()),
        ('title_nostop', TitleNoStop()),
        ('title_lemmatize', TitleLemmatize()),
        ('text_noreuters_nochars', TextNoReutersNoChars()),
        ('text_nopunc', TextNoPunc()),
        ('text_nopropernouns', TextNoProperNouns()),
        ('text_nostop', TextNoStop()),
        ('text_lemmatize', TextLemmatize())
    ])

    title_pipe = Pipeline([
        ('extract_num_letters', FunctionTransformer(extract_num_letters, validate=False)),
        ('extract_num_words', FunctionTransformer(extract_num_words, validate=False)),
        ('extract_perc_up_letters', FunctionTransformer(extract_perc_up_letters, validate=False)),
        ('extract_perc_up_words', FunctionTransformer(extract_perc_up_words, validate=False)),
        ('extract_perc_nums', FunctionTransformer(extract_perc_nums, validate=False)),
        ('extract_mispellings_pol_subj', FunctionTransformer(extract_mispellings_pol_subj, validate=False))
    ])

    text_pipe = Pipeline([
        ('extract_text_features', FunctionTransformer(extract_text_features, validate=False))
    ], verbose=True)

    concat_pipe = Pipeline([
        ('concat_clean_title_text', FunctionTransformer(concat_clean_title_text, validate=False))
    ])

    preprocessing_pipe = Pipeline([
        ('cleaning', clean_pipe),
        ('title_extraction', title_pipe),
        ('text_extraction', text_pipe),
        ('concat_extraction', concat_pipe)
    ])

    new_df = preprocessing_pipe.fit_transform(df)

    return new_df
