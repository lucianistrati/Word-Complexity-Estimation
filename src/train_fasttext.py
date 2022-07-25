from used_repos.personal.aggregated_personal_repos.Word_Complexity_Estimation.dataset_loader import \
    document_preprocess, load_wce_dataset
from gensim.models import FastText

import multiprocessing


def main():
    texts = load_wce_dataset()
    texts = [document_preprocess(text) for text in texts]
    model = FastText(vector_size=256, window=5, min_count=1, sentences=texts, epochs=10,
                     workers=multiprocessing.cpu_count())
    # fasttext language model is trained
    model.save("fasttext.model")  # its checkpoint is then saved


if __name__ == "__main__":
    main()
