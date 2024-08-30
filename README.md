# Word-Complexity-Estimation

## Overview

Word Complexity Estimation is a repository dedicated to the task of assigning a complexity score to words in context within a sentence. This score indicates how difficult it is to understand a word in the given context, with values ranging from 0 to 1. The repository provides tools and models to facilitate the estimation of word complexity based on various features and machine learning techniques.

## Files

- **dataset_loader.py:** Python script for loading and preprocessing the dataset.
- **feature_extractor.py:** Module for extracting features from text data.
- **finetune_xgb.py:** Script for fine-tuning XGBoost model for word complexity estimation.
- **keras_model.py:** Implementation of a Keras model for word complexity estimation.
- **text_preprocess.py:** Utility functions for text preprocessing and cleaning.
- **train_diff_model.py:** Script for training a different model for word complexity estimation.
- **train_fasttext.py:** Script for training a FastText model on the dataset.
- **train_word2vec.py:** Script for training a Word2Vec model on the dataset.
- **utils.py:** Collection of utility functions used across different modules.
- **wce_dataset.npy:** Preprocessed dataset stored in numpy format.
- **xgb_model.py:** Implementation of an XGBoost model for word complexity estimation.

## Workflow

The typical workflow for word complexity estimation using this repository involves the following steps:

1. **Data Loading:** Use `dataset_loader.py` to load and preprocess the dataset.
2. **Feature Extraction:** Extract relevant features from the text data using `feature_extractor.py`.
3. **Model Training:** Train a word complexity estimation model using one of the provided scripts (`finetune_xgb.py`, `keras_model.py`, etc.).
4. **Evaluation:** Evaluate the performance of the trained models using appropriate metrics and techniques.
5. **Prediction:** Use the trained models to predict word complexity scores for new words in context.

## Dataset

The dataset used for training and evaluation is stored in `wce_dataset.npy` and contains preprocessed text data along with corresponding complexity scores.

## Dependencies

Ensure that you have all the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.
