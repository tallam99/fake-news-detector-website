# Data Cleaning
# Fake News Detector EE460J Final Project
# By Tarek Allam

import copy


def get_concat(X):
    return X['concat']


def concat_clean_title_text(X):
    X = copy.deepcopy(X)

    X['concat'] = X['title_lemmatized'] + ' ' + X['text_lemmatized']
    return X
