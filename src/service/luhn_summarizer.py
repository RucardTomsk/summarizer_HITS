import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from text_utils import tokenize, detect_language, custom_sentenize
import numpy as np

# constants
SF_WORD_THRESHOLD = 0.002
SF_SENTENCE_THRESHOLD = 0.3
DEFAULT_TEXT_LANG = "RUSSIAN"


class LuhnExtractiveSummarizer():
    """
    Implementation of the LuhnExtractiveSummarizer  from H.P.Luhn article:
    https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf
    """

    def __init__(self, sf_word_threshold: float = SF_WORD_THRESHOLD,
                 sf_sentence_threshold: float = SF_SENTENCE_THRESHOLD):
        self.sf_word_threshold = sf_word_threshold
        self.sf_sentence_threshold = sf_sentence_threshold
        self._text_lang = DEFAULT_TEXT_LANG

    def tokenize_sent(self, sentences: list) -> list:
        """
        split raw text to tokens
        :param sentences:
        :return: list of tokenized sents
        """

        # tokenization
        tokens = [tokenize(sent, self._text_lang) for sent in sentences]

        return tokens

    def create_word_freq_dict(self, text: str) -> dict:
        """
        create word freq dict for original document
        :param text:
        :return: stemmed word frequence dict for summarized document
        """
        # log(f"tokens:{tokens}")

        tokens = tokenize(text, self._text_lang)
        vectorizer = TfidfVectorizer(use_idf=True)
        X = vectorizer.fit_transform(tokens)
        features_names_out = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame(X.toarray(), columns=features_names_out)  # составляем датафрейм частотностей слов


        freq_dict = {word: np.sum(freq_df[word].values) / len(list(freq_df.keys())) for word in features_names_out}

        freq_dict = dict(sorted(freq_dict.items(), key=lambda y: y[1],
                                reverse=True))  # сортируем словарь частотностей слов по убыванию

        freq_dict = {k: v for k, v in freq_dict.items() if v >= self.sf_word_threshold}

        logging.info(f"Significant words in descending order of frequency: {freq_dict}")

        return freq_dict

    def get_sentence_significance_word_mask(self, sentence_words_mask: list) -> list:
        """
        strip sent by clustering the meaningful part of sentence for proper significance_factor computing
        :param sentence_words_mask:
        :return: sentences_word_mask
        """
        first_sf_word_indx = sentence_words_mask.index(1)
        last_sf_word_indx = len(sentence_words_mask) - 1 - sentence_words_mask[::-1].index(1)

        return sentence_words_mask[first_sf_word_indx: last_sf_word_indx + 1]

    def compute_significance_factor(self, freq_dict: dict, sentence: list) -> np.float16:
        """
        Implementation of significance_factor counting for sentence
        :param freq_dict:
        :param sentence:
        :return: significance_factor for sentence
        """
        sentence_words_mask = [1 if freq_dict[word] else 0 \
                               for word in sentence if word in freq_dict.keys()]

        if sum(sentence_words_mask) == 0:
            return 0

        sentence_words_mask = self.get_sentence_significance_word_mask(sentence_words_mask)
        number_of_sf_words = sum(sentence_words_mask)
        total_number_of_bracketed_words = len(sentence_words_mask)


        significance_factor = number_of_sf_words ** 2 / total_number_of_bracketed_words

        return significance_factor

    def summarize(self, text: str) -> str:
        """
        Creating Extractive Summary from input text
        :param text:
        :return: summary
        """
 

        if self._text_lang == "Unknown":
            logging.info("Unable to detect text language. Check your input file format!!!")
            return ""

        logging.info(f"Input language Detected: {self._text_lang}")

        sentences = custom_sentenize(text)

        text_freq_dict = self.create_word_freq_dict(text)


        sentences_sf = []
        for sent in sentences:
            sentence_tokens = tokenize(sent, self._text_lang)


            sentences_sf.append(
                self.compute_significance_factor(text_freq_dict, sentence_tokens) if len(sentence_tokens) > 0 else 0)


        orig_sentences_sf = sentences_sf.copy()
        sentences_sf.sort(reverse=True)

        sentences_sf_threshold_5 = sentences_sf[1]
        sentences_sf_threshold_10 = sentences_sf[1]


        sentence_sf_threshold_percentile_75 = np.percentile(sentences_sf, 75)


        summary_sentences = [sent for i, sent in enumerate(sentences) if
                             orig_sentences_sf[i] > sentence_sf_threshold_percentile_75]

        print(len(summary_sentences))

        summary = "\n".join(summary_sentences)
        logging.info(f"Final Summary: {summary}")

        return summary


if __name__ == "__main__":
    summarizator = LuhnExtractiveSummarizer()

    test_article = ""
    with open(Path("../../data/netflix_test_article_ru.txt"), "r") as f:
        test_article = "".join(f.readlines())

    print(summarizator.summarize(test_article))
