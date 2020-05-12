# Text Feature Extraction
# Fake News Detector EE460J Final Project
# By Cole Morgan and Menelaos Kaskiris

from textblob import TextBlob
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import re
import copy

sym_spell = None

# Input: The cleaned DataFrame object with column 'text'
# Returns extracted features from article text
def extract_text_features(data):    
    data['text_corrected'] = ""
    data['text_char_count'] = 0
    data['text_word_count'] = 0
    data['text_cap_word_count'] = 0
    data['text_perc_misspells'] = 0
    data['text_polarity'] = 0.0
    data['text_subjectivity'] = 0.0
    for i in range(0, data.shape[0]):
        text = data['text'].iloc[i]
        text_noproper = data['text_noproper'].iloc[i].lower()
        char_count = extract_char_count(text)
        cap_count = extract_cap_word_count(text)
        mpw, text_corr = extract_misspellings(text_noproper)
        word_count = extract_word_count(text)
        pol, subj = extract_polarity_subjectivity(text_corr)

        # Populate feature columns
        data['text_perc_misspells'].iloc[i] = mpw
        data['text_corrected'].iloc[i] = text_corr
        data['text_char_count'].iloc[i] = char_count
        data['text_word_count'].iloc[i] = word_count
        data['text_cap_word_count'].iloc[i] = cap_count
        data['text_polarity'].iloc[i] = pol
        data['text_subjectivity'].iloc[i] = subj
    return data
        

def get_clean_text(data):
    return data['text_lemmatized']


def extract_char_count(s):
    return len(s)


def extract_word_count(s):
    words = re.findall(r'\w+', s)
    n = len(words)
    return n


def extract_cap_word_count(s):
    capital = re.findall(r"\b[A-Z][A-Z]+\b", s)
    return len(capital)


def extract_polarity_subjectivity(s):
    testimonial = TextBlob(s)
    polarity = testimonial.sentiment.polarity
    subjectivity = testimonial.sentiment.subjectivity
    return polarity, subjectivity


def extract_misspellings(s):
    global sym_spell
    if sym_spell is None:
        # Initialize SymSpell checker
        # maximum edit distance per dictionary precalculation
        max_edit_distance_dictionary = 2
        prefix_length = 7
        # create object
        sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

        # load dictionary
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
        # term_index is the column of the term and count_index is the
        # column of the term frequency
        if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                         count_index=1):
            print("Dictionary file not found")

        if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0,
                                                count_index=2):
            print("Bigram dictionary file not found")
            
    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL

    # Start correcting word by word
    article_text = s.split()
    misspelled = 0
    for word in article_text:
        word = word.strip()
        suggestions = sym_spell.lookup(word, suggestion_verbosity, max_edit_distance_lookup)
        # Correct the text
        if len(suggestions) == 0:
            continue
        sug = suggestions[0]
        if sug.term != word:
            s = re.sub("\s+" + word + "\s+", " " + sug.term + " ", s)
            misspelled = misspelled + 1
    mpw = misspelled / len(article_text)

    return mpw, s
