from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from statistics import mean, median
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordsegment import load, segment
from nltk.corpus import wordnet as wn
from wordfreq import word_frequency
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.corpus import wordnet
from textblob import TextBlob

import pronouncing
import inflect
import nltk
import pdb


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
inflect = inflect.engine()
load()


def count_letters(target):
    """

    :param target:
    :return:
    """
    return len(target)


def get_base_word_pct(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    """the higher the more complex a word it is because it requires many subwords"""
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(1 - (len(root) / len(token)))
    if len(ans):
        return mean(ans)
    return ans


def has_prefix(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(not token.startswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_suffix(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(not initial_word.endswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_both_affixes(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(has_prefix(initial_word, root) and has_suffix(initial_word, root))
    if len(ans):
        return mean(ans)
    return ans


def get_base_word_pct_stem(initial_word, root=None, tokens=None):
    """the higher the more complex a word it is because it requires many subwords"""
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(1 - (len(root) / len(initial_word)))
    if len(ans):
        return mean(ans)
    return ans


def has_prefix_stem(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(not initial_word.startswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_suffix_stem(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(not initial_word.endswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_both_affixes_stem(initial_word, root=None, tokens=None):
    """

    :param initial_word:
    :param root:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(initial_word) if tokens is None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(has_prefix_stem(initial_word, root) and has_suffix_stem(initial_word, root))
    if len(ans):
        return mean(ans)
    return ans


def count_hypernyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        word_synsets = wn.synsets(token)
        ans.append(sum([len(word_synset.hypernyms()) for word_synset in word_synsets]))
    if len(ans):
        return mean(ans)
    return ans


def count_hyponyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        word_synsets = wn.synsets(token)
        ans.append(sum([len(word_synset.hyponyms()) for word_synset in word_synsets]))
    if len(ans):
        return mean(ans)
    return ans


def count_antonyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        antonyms_list = []
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms_list.append(lemma.antonyms()[0].name())
        ans.append(len(set(antonyms_list)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_synonyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        synonyms_list = []
        for syn in wordnet.synsets(token):
            for lemma in syn.synonyms():
                synonyms_list.append(lemma.name())
        ans.append(len(set(synonyms_list)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_meronyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.meronyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_part_meroynms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.part_meronyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_substance_meroynms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.substance_meronyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_holonyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.holonyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_part_holonyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.part_holonyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_substance_holonyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.substance_holonyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_entailments(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.entailments()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_troponyms(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.troponyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_average_tokens_length(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        try:
            res = mean([len(word_tokenize(syn.definition())) for syn in wordnet.synsets(token)])
        except:
            res = 0.0
        ans.append(res)
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_average_characters_length(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        try:
            res = mean([len(syn.definition()) for syn in wordnet.synsets(token)])
        except:
            res = 0.0
        ans.append(res)
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_tokens_length(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(word_tokenize(syn.definition())) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_characters_length(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.definition()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def is_singular(word):
    """

    :param word:
    :return:
    """
    return int(inflect.singular_noun(word))


def is_plural(word):
    """

    :param word:
    :return:
    """
    return int(inflect.plural_noun(word))


def check_word_compounding(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(len(segment(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def word_origin(word):
    """

    :param word:
    :return:
    """
    import ety
    origins = ety.origins(word)
    predominant_languages = ["french", "english", "german", "latin", "spanish", "italian", "russian", "greek"]
    mapping = {lang: 0 for lang in predominant_languages}

    for origin in origins:
        for language in predominant_languages:
            if language in origin.language.lower():
                mapping[language] += 1
                break
        mapping[language] = origin.language.lower()

    return max(mapping, key=mapping.get)


def word_polarity(word):
    """

    :param word:
    :return:
    """
    blob = TextBlob(word)
    return blob.sentiment.polarity


def get_target_phrase_ratio(phrase, word):
    """

    :param phrase:
    :param word:
    :return:
    """
    return len(word) / len(phrase)


def get_phrase_len(phrase):
    """

    :param phrase:
    :return:
    """
    return len(phrase)


def get_num_pos_tags(sentence, tokens=None):
    """

    :param sentence:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(sentence) if tokens is None else tokens
    pos_tags = nltk.pos_tag(tokens)
    pos_tags = [pos_tag[1] for pos_tag in pos_tags]
    return len(set(pos_tags)) / len(tokens)


def get_word_position_in_phrase(phrase, start_offset):
    """

    :param phrase:
    :param start_offset:
    :return:
    """
    return start_offset / len(phrase)


def get_phrase_num_tokens(phrase):
    """

    :param phrase:
    :return:
    """
    return len(word_tokenize(phrase))


def custom_wup_similarity(word1, word2):
    """

    :param word1:
    :param word2:
    :return:
    """
    try:
        syn1 = wordnet.synsets(word1)[0]
        syn2 = wordnet.synsets(word2)[0]
        return syn1.wup_similarity(syn2)
    except:
        return 0.0


def get_wup_avg_similarity(target, tokens=None):
    """

    :param target:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(target) if tokens is not None else tokens
    if len(tokens) == 1:
        return 0.0
    else:
        ans = []
        for tok in tokens:
            for tok_ in tokens:
                ans.append(custom_wup_similarity(tok, tok_))
        if len(ans):
            return mean(ans)
        return 0.0


def count_pronounciation_methods(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(len(pronouncing.phones_for_word(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_average_phonemes_per_pronounciation(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        try:
            res = mean([len(pronounciation.split()) for pronounciation in pronouncing.phones_for_word(token)])
        except:
            res = 0.0
        ans.append(res)
    if len(ans):
        return mean(ans)
    return 0.0


def count_total_phonemes_per_pronounciations(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(pronounciation.split()) for pronounciation in pronouncing.phones_for_word(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def get_average_syllable_count(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        pronunciations_list = pronouncing.phones_for_word(token)
        try:
            res = mean([pronouncing.syllable_count(pronunciation_list) for pronunciation_list in pronunciations_list])
        except:
            res = 0.0
        ans.append(res)
    if len(ans):
        return mean(ans)
    return 0.0


def get_total_syllable_count(word, tokens=None):
    """

    :param word:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(word) if tokens is None else tokens
    ans = []
    for token in tokens:
        pronunciations_list = pronouncing.phones_for_word(token)
        ans.append(sum([pronouncing.syllable_count(pronunciation_list) for pronunciation_list in pronunciations_list]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_capital_chars(text):
    """

    :param text:
    :return:
    """
    count = 0
    for i in text:
        if i.isupper():
            count += 1
    return count


def count_capital_words(text, tokens=None):
    """

    :param text:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(text) if tokens is None else tokens
    return sum(map(str.isupper, tokens))


def count_punctuations(text):
    """

    :param text:
    :return:
    """
    punctuations = """}!"#/$%'(*]+,->.:);=?&@\^_`{<|~["""
    res = []
    for i in punctuations:
        res.append(text.count(i))
    if len(res):
        return mean(res)
    return 0.0


def get_word_frequency(target, tokens=None):
    """

    :param target:
    :param tokens:
    :return:
    """
    tokens = word_tokenize(target) if tokens is None else tokens
    return mean([word_frequency(token, "en") for token in tokens])


def main():
    pass


if __name__ == "__main__":
    main()
