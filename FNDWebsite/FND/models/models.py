# Cole's TFIDF Classifier using XGBoost
# TF-IDF Estimator using XGBoost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import os

class TfIdfClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        return None

    def fit_from_file(self, xgb_file, pkl_file):
        bst = xgb.Booster({'nthread': 4})  # init model
        cwd = os.getcwd()
        bst.load_model(cwd + xgb_file)  # load data
        #         self.loaded_ = True
        self.xgb_ = xgb.XGBClassifier()
        self.xgb_._Booster = bst
        pickle = pkl.load(open(cwd + pkl_file, "rb"))
        self.classes_ = pickle[0]
        self.X_ = pickle[1]
        self.y_ = pickle[2]
        self.tfidf_ = pickle[3]
        self.vocab_ = pickle[4]
        self.scaler_ = pickle[5]
        return self

    def save_model(self, xgb_path, pkl_path):
        check_is_fitted(self, attributes=["classes_", "X_", "y_", "tfidf_", "xgb_"])
        self.xgb_.save_model(xgb_path)
        pickle = [self.classes_, self.X_, self.y_, self.tfidf_, self.vocab_, self.scaler_]
        pkl.dump(pickle, open(pkl_path, "wb"))

    def fit(self, X, y):
        # Check that X and y have correct shape
        if X.shape[0] != y.shape[0]:
            print("Not fitted")
            return self
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Learn TF-IDF vocab and extract TF-IDF data
        min_df = 0.001
        max_df = 0.9
        # 81.95 w min_df=0.001, max_df=0.95
        max_features = 380
        # 93.9 w 500

        self.tfidf_ = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features)
        self.scaler_ = StandardScaler(with_mean=False)
        tfidfs = self.tfidf_.fit_transform(X['concat'])
        tfidfs = self.scaler_.fit_transform(tfidfs)
        print(tfidfs.shape)
        vocab = self.tfidf_.vocabulary_
        vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
        #         print(vocab_sorted)
        print(list(w for (w, i) in vocab_sorted))

        self.vocab_ = pd.DataFrame()
        #         print(list(np.sum(np.array(tfidfs), axis=0)))
        self.vocab_['tfidf'] = pd.DataFrame(tfidfs.toarray()).sum(axis=0)
        self.vocab_['word'] = list(w for (w, i) in vocab_sorted)
        self.vocab_ = self.vocab_.sort_values(by='tfidf', axis=0, ascending=False)
        print(self.vocab_)

        #         print(tfidfs)

        # Train XGBoost on TF-IDF data
        # Tune XGBoost using GridSearchCV
        self.xgb_ = xgb.XGBClassifier(random_state=42)
        self.xgb_.fit(tfidfs, y)
        params = {'learning_rate': [0.1, 0.2],
                  'n_estimators': [50, 100, 250, 500, 1000],
                  'max_depth': [2, 4, 6, 8],
                  'random_state': [42]}
        gs = GridSearchCV(xgb.XGBClassifier(), param_grid=params, verbose=10, scoring='roc_auc')
        results = gs.fit(tfidfs, y)
        self.xgb_ = gs.best_estimator_
        print(gs.best_score_)
        print(gs.best_params_)
        # Best params: l_r=0.1, n_estimators=1000, max_depth=8

        self.loaded_ = False

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, attributes=["classes_", "X_", "y_", "tfidf_", "xgb_"])

        tfidfs = self.tfidf_.transform(X['concat'])
        tfidfs = self.scaler_.fit_transform(tfidfs)
        #         if self.loaded_:
        #             return self.xgb_.predict(tfidfs)
        #         else:
        return self.xgb_.predict_proba(tfidfs)[:, 1]

    def getMostFreqWords(self):
        check_is_fitted(self, attributes=["vocab_"])
        return self.vocab_