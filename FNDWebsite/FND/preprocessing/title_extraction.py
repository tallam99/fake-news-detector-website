# Title Feature Extraction
# Fake News Detector EE460J Final Project
# By Jackson Lightfoot and Eduard Alsina

import numpy as np
import copy
from .text_extraction import *


def get_clean_title(X):
    return X['title_lemmatized']


# extraction functions
def extract_num_letters(X):
    X = copy.deepcopy(X)
    X['title_num_letters'] = X['title_nopunc'].apply(str.replace, args=(' ', '')).apply(len)
    return X


def extract_num_words(X):
    X = copy.deepcopy(X)
    X['title_num_words'] = X['title_nopunc'].apply(str.split).apply(len)
    return X


def extract_perc_up_letters(X):
    X = copy.deepcopy(X)

    q = np.array([])
    for title in X['title']:
        q = np.append(q, [sum(c.isupper() for c in title)])

    X['title_perc_up_letters'] = q / X['title'].apply(len)

    return X


def extract_perc_up_words(X):
    X = copy.deepcopy(X)

    perc = np.array([])
    split_titles = X['title_nopunc'].apply(str.split)
    for title in split_titles:
        num_upper = sum(word.isupper() for word in title)
        num_words = len(title)
        perc = np.append(perc, [num_upper / num_words])

    X['title_perc_up_words'] = perc

    return X


def extract_perc_nums(X):
    X = copy.deepcopy(X)

    q = np.array([])
    for title in X['title']:
        q = np.append(q, [sum(c.isdigit() for c in title)])

    X['title_perc_nums'] = q / X['title'].apply(len)

    return X


def extract_mispellings_pol_subj(X):
    X = copy.deepcopy(X)

#     mpw_title_corr = X['title_nopunc'].apply(str.lower).apply(text.extract_misspellings)
#     mpw = mpw_title_corr.apply(lambda x: x[0])
#     title_corr = mpw_title_corr.apply(lambda x: x[1])
#     X['title_perc_misspells'] = mpw
#     X['title_corrected'] = title_corr
#     pol_subj = title_corr.apply(text.extract_polarity_subjectivity)

    pol_subj = X['title_nopunc'].apply(str.lower).apply(extract_polarity_subjectivity)
    pol = pol_subj.apply(lambda x: x[0])
    subj = pol_subj.apply(lambda x: x[1])
    X['title_polarity'] = pol
    X['title_subjectivity'] = subj

    return X
