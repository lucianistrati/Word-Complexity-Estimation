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

numpy_arrays_path = "data/numpy_data"


def create_submission_file(ids, labels, imputer_strategy: str = "max"):
    submission_file_path = "data/submission.txt"

    with open(submission_file_path, 'r+') as f:
        f.truncate(0)

    with open(submission_file_path, "a") as f:
        f.write("id,label\n")
        for id, label in zip(ids, labels):
            if label < 0.0:
                if imputer_strategy == "max":
                    label = max(0.0, label)
                elif imputer_strategy == "mean":
                    label = np.mean(labels)
                elif imputer_strategy == "median":
                    label = np.median(labels)
                else:
                    raise Exception("Unknown imputer strategy")
            f.write(str(id) + "," + str(label) + "\n")


def load_data(embedding_feature: str = "target_word", embedding_model: str = "roberta"):
    X_train_filepath = os.path.join(numpy_arrays_path, "X_train_" + embedding_feature + "_" + embedding_model + ".npy")
    X_train = np.load(file=X_train_filepath, allow_pickle=True)
    y_train_filepath = os.path.join(numpy_arrays_path, "y_train_" + embedding_feature + "_" + embedding_model + ".npy")
    y_train = np.load(file=y_train_filepath, allow_pickle=True)
    X_test_filepath = os.path.join(numpy_arrays_path, "X_test_" + embedding_feature + "_" + embedding_model + ".npy")
    X_test = np.load(file=X_test_filepath, allow_pickle=True)

    y_train = y_train.astype("float32")

    return X_train, y_train, X_test


def load_multiple_models(embedding_models: List[str], embedding_features: List[str], strategy: str = "averaging"):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    for (embedding_model, embedding_feature) in zip(embedding_models, embedding_features):
        X_train, y_train, X_test = load_data(embedding_feature=embedding_feature, embedding_model=embedding_model)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)

    # print([x.shape for x in X_train_list])
    # pdb.set_trace()

    if strategy == "averaging":
        X_train_list = np.array(X_train_list)
        X_test_list = np.array(X_test_list)
        X_train = np.mean(X_train_list, axis=0)
        y_train = y_train_list[0]
        X_test = np.mean(X_test_list, axis=0)
    elif strategy == "stacking":
        X_train = np.hstack(X_train_list)
        y_train = y_train_list[0]
        X_test = np.hstack(X_test_list)
    elif strategy == "ensemble":
        return X_train_list, y_train_list, X_test_list
    elif strategy == "dimensionality_reduction":
        reduction_method = "PCA"  # "LDA", "TSNE"
        if reduction_method == "PCA":
            dimensionality_reducer = PCA()
        elif reduction_method == "LDA":
            dimensionality_reducer = LDA()
        elif reduction_method == "TSNE":
            dimensionality_reducer = TSNE()
        for (X_train, X_test) in zip(X_train_list, X_test_list):
            X_train = dimensionality_reducer.fit_transform(X_train, n_components=min(X_train.shape))
            X_test = dimensionality_reducer.transform(X_test, n_components=min(X_test.shape))
        X_train = np.hstack(X_train_list)
        y_train = y_train_list[0]
        X_test = np.hstack(X_test_list)
    return X_train, y_train, X_test


def main():
    pass


if __name__ == "__main__":
    main()
