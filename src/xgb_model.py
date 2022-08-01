from used_repos.personal.aggregated_personal_repos.Word_Complexity_Estimation.src.feature_extractor import \
    check_word_compounding, count_antonyms, count_average_phonemes_per_pronounciation, count_capital_chars, \
    count_capital_words, count_definitions_average_characters_length, count_definitions_average_tokens_length, \
    count_definitions_characters_length, count_definitions_tokens_length, count_entailments, \
    count_holonyms, count_hypernyms, count_hyponyms, count_letters, count_meronyms, count_part_holonyms, \
    count_part_meroynms, count_pronounciation_methods, count_punctuations, count_substance_holonyms, \
    count_substance_meroynms, count_synonyms, count_total_phonemes_per_pronounciations, count_troponyms, \
    custom_wup_similarity, get_average_syllable_count, get_base_word_pct, get_base_word_pct_stem, get_num_pos_tags, \
    get_phrase_len, get_phrase_num_tokens, get_target_phrase_ratio, get_total_syllable_count, get_word_frequency, \
    get_word_position_in_phrase, get_wup_avg_similarity, has_both_affixes, has_both_affixes_stem, has_prefix, \
    has_prefix_stem, has_suffix, has_suffix_stem, is_plural, is_singular, mean, median, word_frequency, \
    word_origin, word_polarity, word_tokenize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from used_repos.personal.aggregated_personal_repos.Word_Complexity_Estimation.src.dataset_loader import document_preprocess
from transformers import RobertaTokenizer, RobertaModel, pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from keras.models import Sequential
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
from keras.layers import Dense
from pandas import read_csv
from sklearn.svm import SVR
from gensim import corpora
from copy import deepcopy
from typing import List
from tqdm import tqdm

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np

import textstat
import linalg
import torch
import nltk
import copy
import csv
import pdb
import os
from used_repos.personal.aggregated_personal_repos.Word_Complexity_Estimation.src.finetune_xgb import finetune_xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge, HuberRegressor


def train_basic_model(X_train, y_train, X_test, embedding_feature: str = "target_word", embedding_model: str = "roberta"):
    data_scaler = StandardScaler()
    label_scaler = StandardScaler()
    regressor = XGBRegressor(eval_metric=mean_absolute_error, max_depth=10, min_child_weight=1)

    def cross_val_func(regressor, X_train, y_train):
        kfolder5 = KFold(n_splits=5, shuffle=False)
        print(X_train.shape)
        scores = cross_val_score(regressor, X_train, y_train, scoring="neg_mean_absolute_error", cv=kfolder5, n_jobs=-1)
        return [score * (-1) for score in scores]

    regressors_list = [RandomForestRegressor(random_state=100), LinearRegression(), SVR()]
    regressor_names = ["random forest", "linear regressor", "svr"]
    for (regressor, name) in list(zip(regressors_list, regressor_names)):
        print(name, ":", cross_val_func(regressor, X_train, y_train))

    print(X_train.shape, X_test.shape, X_train.dtype, X_test.dtype)

    X_train = data_scaler.fit_transform(X_train)
    X_test = data_scaler.transform(X_test)

    y_train = label_scaler.fit_transform(np.reshape(y_train, (y_train.shape[0], 1)))

    regressor.fit(X_train, y_train)
    print("Embedding feature:", embedding_feature)
    print("Embedding model:", embedding_model)

    # finetune_xgb(X_train, y_train, X_test, label_scaler)
    # y_pred = np.clip(y_pred, a_min=0.0, a_max=1.0)
    # print("MAE:",  mean_absolute_error(y_validation, y_pred))
    return label_scaler.inverse_transform(regressor.predict(X_test))


def main():
    pass


if __name__ == "__main__":
    main()
