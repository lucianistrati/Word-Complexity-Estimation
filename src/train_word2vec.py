from used_repos.personal.Word_Complexity_Estimation.src.embeddings_train.dataset_loader import load_wce_dataset
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import multiprocessing
import nltk
import pdb


def document_preprocess(document):
    return word_tokenize(document)


def main():
    texts = load_wce_dataset()

    texts = [document_preprocess(text) for text in texts]

    print(multiprocessing.cpu_count())

    print(texts[0])
    print(texts[-1])
    model = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=multiprocessing.cpu_count())
    model.save("word2vec.model")  # checkpoint is saved
    print("saved")


if __name__ == "__main__":
    main()
