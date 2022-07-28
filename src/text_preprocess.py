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
from transformers import AlbertTokenizer, TFAlbertModel
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, TFBertModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from used_repos.personal.aggregated_personal_repos.Word_Complexity_Estimation.src.dataset_loader import document_preprocess
from transformers import pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.tokenize import TreebankWordTokenizer as twt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
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


textstat.set_lang("en")
stop_words = set(stopwords.words('english'))
PAD_TOKEN = "__PAD__"

try:
    word2vec_model = Word2Vec.load("src/embeddings_train/word2vec.model")
except FileNotFoundError:
    print("No word2vec model in checkpoints!")

numpy_arrays_path = "data/numpy_data"
try:
    word2vec_model = Word2Vec.load("src/embeddings_train/fasttext.model")
except FileNotFoundError:
    print("No fasttext model in checkpoints!")

# def document_preprocess(document):
#     return document.lower().split()

# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model")
# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model.syn1neg.npy")
# word2vec_model = Word2Vec.load("src/embeddings_train/abcnews_word2vec.model.wv.vectors.npy")


def mask_expression(text, start_offset, end_offset):
    return text[:start_offset] + "<mask>" + text[end_offset:]


def predict_masked_tokens(text):
    unmasker = pipeline('fill-mask', model='roberta-base')
    return unmasker(text)[0]["token_str"]


def embed_text(text, embedding_model: str = "word2vec_trained", phrase="", start_offset="", end_offset=""):
    """This function embedds the text given using an embedding model, it might also use the phrase, start_offset or the
    end_offset for certain embeddings"""
    if embedding_model == "roberta":
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = RobertaModel.from_pretrained("roberta-base")
        encoded_input = roberta_tokenizer(text, return_tensors="pt")
        output = roberta_model(**encoded_input)
        return torch.reshape(output.pooler_output, shape=(output.pooler_output.shape[1],)).detach().numpy()
    elif embedding_model == "sentence_transformer":
        sent_transf_model = SentenceTransformer("all-mpnet-base-v2")
        embedding = sent_transf_model.encode(text)
        return embedding
    elif embedding_model == "sentence_transformer_multi_qa":
        sent_transf_model_multi_qa = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        embedding = sent_transf_model_multi_qa.encode(text)
        return embedding
    elif embedding_model == "roberta_large":
        roberta_large_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        roberta_large_model = RobertaModel.from_pretrained("roberta-large")
        encoded_input = roberta_large_tokenizer(text, return_tensors="pt")
        output = roberta_large_model(**encoded_input)
        return torch.reshape(output.pooler_output, shape=(output.pooler_output.shape[1],)).detach().numpy()
    elif embedding_model == "bert_large":
        bert_large_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        bert_large_model = TFBertModel.from_pretrained("bert-large-uncased")
        encoded_input = bert_large_tokenizer(text, return_tensors="tf")
        output = bert_large_model(**encoded_input)
        return np.reshape(np.array(output.pooler_output), newshape=(output.pooler_output.shape[1],))
    elif embedding_model == "albert-base-v2":
        albert_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        albert_model = TFAlbertModel.from_pretrained("albert-base-v2")
        encoded_input = albert_tokenizer(text, return_tensors="tf")
        output = albert_model(**encoded_input)
        return np.reshape(np.array(output.pooler_output), newshape=(output.pooler_output.shape[1],))
    elif embedding_model == "roberta_distil":
        sent_transf_model_distil_roberta = SentenceTransformer("all-distilroberta-v1")
        embedding = sent_transf_model_distil_roberta.encode(text)
        return embedding
    elif embedding_model == "all_minimlm_l12":
        all_minimlm_l12_model = SentenceTransformer("all-MiniLM-L12-v2")
        embedding = all_minimlm_l12_model.encode(text)
        return embedding
    elif embedding_model == "all_minimlm_l6":
        all_minimlm_l6_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = all_minimlm_l6_model.encode(text)
        return embedding
    elif embedding_model.startswith("tfidfvectorizer"):
        return text
    elif embedding_model.startswith("gensim_doc2bow"):
        return text.split()
    elif embedding_model == "word2vec_trained":
        try:
            vector = word2vec_model.wv[document_preprocess(text)]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector
    elif embedding_model == "paper_features":
        return get_paper_features(phrase, text, start_offset, end_offset), None
    elif embedding_model == "word2vec_trained_special":
        try:
            vector = word2vec_model.wv[text]
            vector = np.mean(vector, axis=0)
            vector = np.reshape(vector, (1, vector.shape[0]))
            print(vector.shape)
        except KeyError:
            vector = np.random.rand(1, 300)
        return vector


def embed_data(embedding_feature: str, embedding_model: str):
    """This function embedds the data based on a chosen feature to be emebedded/chosen set of rules to be applied
    over the feature with a certain alias and then an embedding model might be used to pursue the feature extraction"""
    train_path = "data/train_full.txt"
    test_path = "data/test.txt"
    if embedding_feature == "phrase":
        column_idx = 1
    elif embedding_feature == "target_word":
        column_idx = 4

    train_columns = ["id", "phrase", "start_offset", "end_offset", "target_word",
                     "native_annotators", "non_native_annotators", "difficult_native_annotators",
                     "difficult_non_native_annotators", "label"]
    test_columns = ["id", "phrase", "start_offset", "end_offset", "target_word",
                    "native_annotators", "non_native_annotators"]

    print(len(train_columns) == len(test_columns))

    print("Embedding feature:", embedding_feature)

    X_train, y_train, X_test = [], [], []

    datapoints_limit = -1

    X_train_str = []
    X_test_str = []
    with open(train_path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        for i, row in tqdm(enumerate(data)):
            if datapoints_limit != -1:
                if i > datapoints_limit:
                    break
            if embedding_feature == "phrase_with_no_target_word":
                start_offset = int(row[2])
                end_offset = int(row[3])
                X_train.append(embed_text(row[1][:start_offset] + row[1][end_offset:], embedding_model=embedding_model))
            elif embedding_feature in ["phrase", "target_word"]:
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                outputs = embed_text(row[column_idx], embedding_model=embedding_model, phrase=phrase,
                                     start_offset=start_offset, end_offset=end_offset)
                if embedding_model == "paper_features":
                    X_train.append(np.array(outputs[0]))
                    if outputs[1] is not None:
                        X_train_str.append(outputs[1])
                else:
                    X_train.append(outputs)
            elif embedding_feature == "predicted_target_word":
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                masked_phrase = mask_expression(phrase, start_offset, end_offset)
                masked_prediction = predict_masked_tokens(masked_phrase)
                X_train.append(embed_text(masked_prediction, embedding_model=embedding_model))
            elif embedding_feature == "phrase_special_tokenized":
                start_offset = int(row[2])
                end_offset = int(row[3])
                text_ = row[1][:start_offset].lower().split() + [row[4].lower()] + row[1][end_offset:].lower().split()
                X_train.append(embed_text(text_, embedding_model=embedding_model))
            y_train.append(float(row[8]) / float(row[6]))

    X_train, y_train = np.array(X_train), np.array(y_train)

    y_train_filename = "y_train_" + embedding_feature + "_" + embedding_model + "_non_native.npy"
    y_train_filepath = os.path.join(numpy_arrays_path, y_train_filename)
    np.save(file=y_train_filepath, arr=y_train)

    with open(test_path) as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        for i, row in tqdm(enumerate(data)):
            if datapoints_limit != -1:
                if i > datapoints_limit:
                    break
            if embedding_feature == "phrase_with_no_target_word":
                start_offset = int(row[2])
                end_offset = int(row[3])
                X_test.append(embed_text(row[1][:start_offset] + row[1][end_offset:], embedding_model=embedding_model))
            elif embedding_feature in ["phrase", "target_word"]:
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                outputs = embed_text(row[column_idx], embedding_model=embedding_model, phrase=phrase,
                                     start_offset=start_offset, end_offset=end_offset)
                # vecs = []
                # res_types = []
                # final_vecs = []
                # for word in word_tokenize(row[column_idx]):
                #     res_type, vec = get_embedding_word(word)
                #     res_types.append(res_type)
                #     vecs.append(vec)
                # if "good" in res_types:
                #     for res_type in res_types:
                #         if res_type == "good":
                #             final_vecs.append(vec)
                # if len(final_vecs):
                #     vecs = final_vecs
                # vecs = np.mean(vecs, axis=0)
                # assert vecs.shape == (1,10)
                if embedding_model == "paper_features":
                    # for elem in vecs[0]:
                    #     outputs[0].append(elem)
                    X_test.append(np.array(outputs[0], dtype=np.float))
                    if outputs[1] is not None:
                        X_test_str.append(outputs[1])
                else:
                    X_test.append(outputs)
            elif embedding_feature == "predicted_target_word":
                phrase = row[1]
                start_offset = int(row[2])
                end_offset = int(row[3])
                masked_phrase = mask_expression(phrase, start_offset, end_offset)
                masked_prediction = predict_masked_tokens(masked_phrase)
                X_test.append(embed_text(masked_prediction, embedding_model=embedding_model))
            elif embedding_feature == "phrase_special_tokenized":
                text_ = row[1][:start_offset].lower().split() + [row[4].lower()] + row[1][end_offset:].lower().split()
                X_test.append(embed_text(text_, embedding_model=embedding_model))

    X_test = np.array(X_test)

    # corpus = [x for x in X_train] + [x for x in X_test]
    # corpus = list(set(corpus))
    # print(len(corpus))
    # print(corpus[0])
    # print(corpus[-1])
    # np.save(file="data/wce_dataset.npy", arr=np.array(corpus), allow_pickle=True)
    # exit(0)

    print(X_train.shape, X_test.shape, "PRE CONCAT WITH STR FEATURES")
    if len(X_test_str) and len(X_train_str):
        cv = TfidfVectorizer(analyzer='char')
        X_train_str = cv.fit_transform(X_train_str).toarray()
        X_test_str = cv.transform(X_test_str).toarray()

        print(X_train_str.shape, X_test_str.shape)

        X_train_str = X_train_str.astype(float)
        X_test_str = X_test_str.astype(float)

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        print(X_train_str.dtype)
        print(X_test_str.dtype)

        print(X_train.dtype)
        print(X_test.dtype)

        X_train = np.hstack((X_train, X_train_str))
        X_test = np.hstack((X_test, X_test_str))

    print(X_train.shape, X_test.shape, "POST CONCAT WITH STR FEATURES")

    tfidfvectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4))

    if embedding_model.startswith("tfidfvectorizer"):
        X_train = tfidfvectorizer.fit_transform(X_train).toarray()
        X_test = tfidfvectorizer.transform(X_test).toarray()
    elif embedding_model == "gensim_doc2bow":
        corpus = [x for x in X_train] + [x for x in X_test]
        dict_d = corpora.Dictionary(corpus)

        X_train = [dict_d.doc2bow(x) for x in X_train]  # one list of tuples of two integers
        X_train = np.array([np.array(x) for x in X_train])
        X_train = np.array([np.linalg.norm(x) for x in X_train])  # one float per sentence/word

        X_test = [np.array(dict_d.doc2bow(x)) for x in X_test]  # one list of tuples of two integers
        X_test = np.array([np.array(x) for x in X_test])
        X_test = np.array([np.linalg.norm(x) for x in X_test])  # one float per sentence/word

    np.save(file=os.path.join(numpy_arrays_path, "X_train_" + embedding_feature + "_" + embedding_model + ".npy"), arr=X_train)
    np.save(file=os.path.join(numpy_arrays_path, "y_train_" + embedding_feature + "_" + embedding_model + ".npy"), arr=y_train)
    np.save(file=os.path.join(numpy_arrays_path, "X_test_" + embedding_feature + "_" + embedding_model + ".npy"), arr=X_test)

    return X_train, y_train, X_test


def count_word_senses(word, tokens=None):
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(len(wn.synsets(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_vowels(word):
    return len([c for c in word if c in "aeiou"])


def count_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    return len([c for c in word if c in consonants])


def count_double_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    cnt = 0
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and word[i] in consonants and word[i + 1] in consonants:
            cnt += 1
    return cnt


def get_double_consonants_pct(word):
    return count_double_consonants(word) / len(word)


def get_vowel_pct(word):
    return count_vowels(word) / len(word)


def get_consonants_pct(word):
    return count_consonants(word) / len(word)


def get_part_of_speech(sentence, tokens=None):
    tokens = word_tokenize(sentence) if tokens is None else tokens
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([pos_tag[1] for pos_tag in pos_tags])


def get_good_vectorizer():
    return TfidfVectorizer(analyzer='char_wb', n_gram_range=(1, 4))


def spans(phrase):
    return list(twt().span_tokenize(phrase))


def count_sws(text, tokens=None):
    if tokens is None:
        tokens = word_tokenize(text)
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(text):
    tokens = word_tokenize(text)
    return count_sws(text, tokens) / len(tokens)


def get_context_tokens(phrase, start_offset, end_offset, context_size=1):  # try 2
    tokens = [PAD_TOKEN for _ in range(context_size)] + nltk.word_tokenize(phrase) + [PAD_TOKEN for _ in range(context_size)]
    tokens_spans = [(0, 0) for _ in range(context_size)] + spans(phrase) + [(0, 0) for _ in range(context_size)]
    for i, (l, r) in enumerate(tokens_spans):
        if r >= start_offset:
            return tokens[i - context_size: i] + tokens[i + 1: i + context_size + 1]
    return None


def embed_multiple_models(embedding_models: List[str], embedding_features: List[str], strategy: str = "averaging"):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    for (embedding_model, embedding_feature) in zip(embedding_models, embedding_features):
        X_train, y_train, X_test = embed_data(embedding_feature=embedding_feature, embedding_model=embedding_model)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)

    print([x.shape for x in X_train_list])
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


GOOD = 0
ERRORS = 0


def get_paper_features(phrase, target, start_offset, end_offset):
    """
    :param phrase:
    :param target:
    :param start_offset:
    :param end_offset:
    :return:
    """
    context_tokens = get_context_tokens(phrase, start_offset, end_offset)
    if context_tokens is None:
        context_tokens = []

    num_features = []
    word = target
    num_features_ = [count_letters(target),
                     count_consonants(target),
                     count_vowels(target),
                     get_vowel_pct(target),
                     get_consonants_pct(target),
                     get_double_consonants_pct(target),
                     count_word_senses(target, tokens=word_tokenize(target)),
                     mean([count_word_senses(context_tok) for context_tok in context_tokens]),
                     get_base_word_pct(target, tokens=word_tokenize(word)),
                     has_suffix(target, tokens=word_tokenize(word)),
                     count_letters(target),
                     get_base_word_pct_stem(target, tokens=word_tokenize(word)),
                     has_both_affixes_stem(target, tokens=word_tokenize(word)),
                     count_hypernyms(target, tokens=word_tokenize(word)),
                     count_hyponyms(target, tokens=word_tokenize(word)),
                     count_antonyms(target, tokens=word_tokenize(word)),
                     count_definitions_average_tokens_length(target, tokens=word_tokenize(word)),
                     count_definitions_average_characters_length(target, tokens=word_tokenize(word)),
                     count_definitions_tokens_length(target, tokens=word_tokenize(word)),
                     count_total_phonemes_per_pronounciations(target, tokens=word_tokenize(word)),
                     get_word_frequency(target, tokens=word_tokenize(word)),
                     get_average_syllable_count(target),
                     check_word_compounding(target),
                     get_base_word_pct(target),
                     ]
    for feature in num_features_:
        num_features.append(feature)
    # num_features.append(textstat.flesch_reading_ease(target))
    # num_features.append(textstat.flesch_kincaid_grade(target))
    # num_features.append(textstat.smog_index(target))
    # num_features.append(textstat.coleman_liau_index(target))
    # num_features.append(textstat.automated_readability_index(target))
    # num_features.append(textstat.dale_chall_readability_score(target))
    # num_features.append(textstat.difficult_words(target))
    # num_features.append(textstat.linsear_write_formula(target))
    # num_features.append(textstat.gunning_fog(target))

    pos_features = get_part_of_speech(target)
    print(len(pos_features))

    return num_features


def main():
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])

    for synset in wn.synsets('flexed'):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())
    # print(get_context_tokens(phrase, start_offset, end_offset))


if __name__ == '__main__':
    main()
