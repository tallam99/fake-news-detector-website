# Data Cleaning
# Fake News Detector EE460J Final Project
# By Tarek Allam

from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import copy

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stops = set(stopwords.words('english'))
punc = [',', ';', '.', 'a', 'i', 'I', 'A']
reuters_map = ['Reuters', 'reuters', 'Reuters,', 'reuters,', 'Reuters;', 'reuters;', 'Reuters.', 'reuters.', 'MOSCOW,']
video_map = ["[Video]", "[VIDEO]", "[video]", "(Video)", "(VIDEO)", "(video)"]


# TODO: May want to filter all cap locations at beginning of some articles (mostly real ones) Ex: WASHINGTON, KUALA LUMPUR


# Block 1
class TitleNoVideoNoReuters(BaseEstimator, TransformerMixin):
    def remove_video(self, X):
        no_video = ""
        X = str(X)

        for word in X.split():
            if word in video_map:
                continue
            else:
                no_video += word + " "

        return no_video

    def remove_reuters(self, X):
        no_reuters = ""
        X = str(X)

        for word in X.split():
            if word in reuters_map:
                continue
            else:
                no_reuters += word + " "

        return no_reuters

    def transform(self, X, *_):
        X['title_novideo_noreuters'] = X['title'].map(self.remove_reuters).map(self.remove_video)
        return X

    def fit(self, *_):
        return self


# Block 2
class TitleNoPunc(BaseEstimator, TransformerMixin):
    def remove_punc(self, X):
        no_punc = ""
        for word in X.split():
            no_punc += ''.join(c for c in word if c.isalpha()) + " "

        return no_punc

    def transform(self, X, *_):
        X['title_nopunc'] = X['title_novideo_noreuters'].map(self.remove_punc)
        return X

    def fit(self, *_):
        return self


# Block 3
class TitleNoStop(BaseEstimator, TransformerMixin):
    def remove_stop(self, X):
        no_stop = ""
        for word in X.split():
            if word not in stops:
                no_stop += word + " "

        return no_stop

    def transform(self, X, *_):
        X['title_nostop'] = X['title_nopunc'].map(self.remove_stop)
        return X

    def fit(self, *_):
        return self


# Block 4
class TitleLemmatize(BaseEstimator, TransformerMixin):
    # function to convert nltk tag to wordnet tag
    def nltk_tag_to_wordnet_tag(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, X):
        X = re.sub(r'[0-9]+', '', str(X))
        lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(X))
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized = ""
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized += word + " "
            else:
                lemmatized += lemmatizer.lemmatize(word, tag) + " "

        return lemmatized

    def transform(self, X, *_):
        X['title_lemmatized'] = X['title_nostop'].apply(str.lower).map(self.lemmatize)
        return X

    def fit(self, *_):
        return self


# Block 5
class TextNoReutersNoChars(BaseEstimator, TransformerMixin):
    def remove_reuters_chars(self, X):
        no_reuters = ""
        X = str(X)

        for word in X.split():
            if len(word) == 1 and word not in punc:
                continue

            word = ''.join(c for c in word if c.isalnum() or c in punc)
            if word in reuters_map:
                continue
            no_reuters += word + " "

        return no_reuters

    def transform(self, X, *_):
        X['text_noreuters_nochars'] = X['text'].map(self.remove_reuters_chars)
        return X

    def fit(self, *_):
        return self


# Block 6
class TextNoPunc(BaseEstimator, TransformerMixin):
    def remove_punc(self, X):
        no_punc = ""
        for word in X.split():
            no_punc += ''.join(c for c in word if c.isalnum()) + " "

        return no_punc

    def transform(self, X, *_):
        X['text_nopunc'] = X['text_noreuters_nochars'].map(self.remove_punc)
        return X

    def fit(self, *_):
        return self


# Block 7
class TextNoProperNouns(BaseEstimator, TransformerMixin):
    def remove_prop(self, X):
        data = copy.deepcopy(X)
        match = re.search("(\w,*)\s+[A-Z][A-Za-z]+", data)
        while match is not None:
            data = data.replace(match.group(), match.group(1) + " ")
            match = re.search("(\w,*)\s+[A-Z][A-Za-z]+", data)
        return data

    def transform(self, X, *_):
        X['text_noproper'] = X['text_nopunc'].map(self.remove_prop)
        return X

    def fit(self, *_):
        return self


# Block 8
class TextNoStop(BaseEstimator, TransformerMixin):
    def remove_stop(self, X):
        no_stop = ""
        for word in X.split():
            if word not in stops:
                no_stop += word + " "

        return no_stop

    def transform(self, X, *_):
        X['text_nostop'] = X['text_noproper'].map(self.remove_stop)
        return X

    def fit(self, *_):
        return self


# Block 9
class TextLemmatize(BaseEstimator, TransformerMixin):
    # function to convert nltk tag to wordnet tag
    def nltk_tag_to_wordnet_tag(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, X):
        X = re.sub(r'[0-9]+', '', str(X))
        lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(X))
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized = ""
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized += word + " "
            else:
                lemmatized += lemmatizer.lemmatize(word, tag) + " "

        return lemmatized

    def transform(self, X, *_):
        X['text_lemmatized'] = X['text_nostop'].apply(str.lower).map(self.lemmatize)
        X = X.dropna()
        return X

    def fit(self, *_):
        return self
