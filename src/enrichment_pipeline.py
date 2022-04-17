import re
import pke
# import nltk
# import math
import spacy
import textacy
import logging
import warnings
import textstat
# import numpy as np
import contractions
from typing import Tuple, List
# from transformers import pipeline
from collections import deque, Counter
from textacy.extract import keyterms as kt
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings('ignore')

# classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-distilroberta-base')


# def zero_shot_classify(text: str, candidate_labels: list, multi_class: bool = False):
#     result = classifier(text, candidate_labels, multi_label=multi_class)
#     return result['labels'][np.argmax(result['scores'])]


# class SentimentExtractor:
#     def __init__(self):
#         self.__text = ""
#         self.__sentences = None
#         self.__sentiment_labels = ['positive', 'neutral', 'negative']
#         self.__bottom_percent = 0.2
#         self.__short_sentence_length = 2
#         self.__large_sentence_length = 100

#     def remove_short_and_lengthy_sentences(self):
#         x = []
#         for sent in self.__sentences:
#             if len(sent.split()) < self.__short_sentence_length:
#                 continue
#             elif len(sent.split()) > self.__large_sentence_length:
#                 continue
#             else:
#                 x.append(sent)
#         return x

#     def get_bottom_sentences(self):
#         b_count = math.floor(len(self.__sentences) * self.__bottom_percent)
#         sentences_q = deque(self.__sentences)
#         bottom_sentences = [sentences_q.pop() for _ in range(b_count)]
#         return bottom_sentences

#     def get_sentence_list_sentiments(self, sentence_list: list):

#         sentence_sentiments = []
#         for index, sentence in enumerate(sentence_list):
#             item = {'sentence': sentence,
#                     'sentiment': zero_shot_classify(text=sentence, candidate_labels=self.__sentiment_labels)
#                     }
#             sentence_sentiments.append(item)
#         return sentence_sentiments

#     def extract(self, text: str):
#         self.__text = text
#         logging.info(f'Extracting Document Sentiment')
#         self.__sentences = nltk.sent_tokenize(self.__text)
#         self.__sentences = self.remove_short_and_lengthy_sentences()
#         bottom_sentences = self.get_bottom_sentences()
#         bottom_sentiments = self.get_sentence_list_sentiments(bottom_sentences)
#         bottom_sentiment = SentimentExtractor.get_major_sentiment(bottom_sentiments)
#         return "neutral" if len(bottom_sentiment) == 0 else bottom_sentiment[0][0]

#     @staticmethod
#     def get_major_sentiment(sentence_sentiment_list: list):
#         return Counter([item['sentiment'] for item in sentence_sentiment_list]).most_common(1)


class StatisticsExtractor:
    def __init__(self, stopwords: list):
        self.__text = None
        self.__stop_words = stopwords

    def get_total_sentences(self):
        return textstat.sentence_count(self.__text)

    def get_total_words(self):
        return textstat.lexicon_count(self.__text, removepunct=True)

    def get_total_characters(self):
        return textstat.char_count(self.__text, ignore_spaces=True)

    def get_average_word_length(self):
        w_len = [len(w) for w in self.__text.split(" ")]
        return round(sum(w_len) / len(w_len), 2)

    def get_average_sentence_length(self):
        s_len = [len(s) for s in sent_tokenize(self.__text)]
        return round(sum(s_len) / len(s_len), 2)

    def get_paragraph_count(self, min_para_word_count=15):
        paras = self.__text.split("\n\n")
        true_paras = [p for p in paras if len(p.split()) > min_para_word_count]
        return len(true_paras)

    def get_syllable_count(self):
        words = word_tokenize(self.__text)
        s_count = [textstat.syllable_count(w) for w in words]
        s_count = sum(s_count) / len(s_count)
        return round(s_count, 2)

    def top_n_most_frequent_words(self, n=5):
        # Remove stopwords and check the count of each word, get the top n
        words = self.__text.split(" ")
        words = [w.lower() for w in words if w.lower() not in self.__stop_words]
        word_counter = Counter(words)
        high_freq_words = [w for w, freq in word_counter.most_common(n)]
        return high_freq_words

    def get_reading_difficulty(self):
        return textstat.flesch_reading_ease(self.__text)

    def extract(self, text: str):
        self.__text = text
        logging.info(f'Extracting Text Statistics')
        result = {
            "paragraph_count": self.get_paragraph_count(),
            "sentence_count": self.get_total_sentences(),
            "word_count": self.get_total_words(),
            "character_count": self.get_total_characters(),
            "avg_word_length": self.get_average_word_length(),
            "avg_sentence_length": self.get_average_sentence_length(),
            "syllable_count": self.get_syllable_count(),
            "frequent_words": self.top_n_most_frequent_words(),
            "reading_difficulty": self.get_reading_difficulty(),
        }
        return result


class KeywordExtractor:
    def __init__(self, stopwords: list, spacy_lang_model):
        self.__text = ""
        self.__stopwords = stopwords
        self.__spacy_model = spacy_lang_model
        self.__topic_rank = pke.unsupervised.TopicRank()
        self.__text_rank = pke.unsupervised.TextRank()
        self.__yake = pke.unsupervised.YAKE()
        self.__single_rank = pke.unsupervised.SingleRank()
        self.__position_rank = pke.unsupervised.PositionRank()
        self.__multi_partite = pke.unsupervised.MultipartiteRank()

    def preprocess(self):
        self.__text = contractions.fix(self.__text)
        self.__text = re.sub(r'\n\s*\n', '\n\n', self.__text)

    def topic_rank_pke_extraction(self, text=None):
        input_text = self.__text if text is None else text
        extraction_model = self.__topic_rank
        extraction_model.load_document(input=input_text, language='en', normalization='lemmatization')
        extraction_model.candidate_selection(pos={'NOUN', 'PROPN'})
        extraction_model.candidate_weighting(threshold=0.80)
        key_phrases = extraction_model.get_n_best(n=5, stemming=True)
        return [item[0] for item in key_phrases]

    def text_rank_pke_extraction(self, text=None):
        input_text = self.__text if text is None else text
        extractor = self.__text_rank
        extractor.load_document(input=input_text, language='en', normalization='lemmatization')
        extractor.candidate_weighting(window=2, pos={'NOUN', 'PROPN'}, top_percent=0.80)
        key_phrases = extractor.get_n_best(n=3, stemming=True)
        return [item[0] for item in key_phrases]

    def yake_pke_extraction(self, text=None):
        input_text = self.__text if text is None else text
        extractor = self.__yake
        extractor.load_document(input=input_text, language='en', normalization='lemmatization')
        extractor.candidate_selection(n=3)
        extractor.candidate_weighting(window=2,
                                      use_stems=True)
        key_phrases = extractor.get_n_best(n=3, threshold=0.8)
        return [item[0] for item in key_phrases]

    def single_rank_pke_extraction(self, text=None):
        input_text = self.__text if text is None else text
        extractor = self.__single_rank
        extractor.load_document(input=input_text, language='en', normalization='lemmatization')
        extractor.candidate_selection(pos={'NOUN', 'PROPN'})
        extractor.candidate_weighting(window=5, pos={'NOUN', 'PROPN'})
        key_phrases = extractor.get_n_best(n=5)
        return [item[0] for item in key_phrases]

    def position_rank_pke_extraction(self, text=None):
        input_text = self.__text if text is None else text
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        extractor = self.__position_rank
        extractor.load_document(input=input_text, language='en', normalization='lemmatization')
        extractor.candidate_selection(grammar=grammar, maximum_word_number=3)
        extractor.candidate_weighting(window=5, pos={'NOUN', 'PROPN'})
        key_phrases = extractor.get_n_best(n=5)
        return [item[0] for item in key_phrases]

    def multi_partite_rank_pke_extraction(self, text=None):
        input_text = self.__text if text is None else text
        extractor = self.__multi_partite
        extractor.load_document(input=input_text)
        extractor.candidate_selection(pos={'NOUN', 'PROPN'})
        extractor.candidate_weighting(alpha=1.1, threshold=0.80, method='average')
        key_phrases = extractor.get_n_best(n=5)
        return [item[0] for item in key_phrases]

    def textacy_graph_extraction(self, text=None):
        input_text = self.__text if text is None else text
        doc = textacy.make_spacy_doc(input_text, lang=self.__spacy_model)
        sg_rank = [kw for kw, wt in kt.sgrank(doc, ngrams=(1, 2, 3, 4), normalize="lemma", topn=5)]
        s_cake = [kw for kw, wt in kt.scake(doc, normalize="lemma", topn=5)]
        return [list(item)[0] for item in textacy.extract.utils.aggregate_term_variants(set(sg_rank + s_cake))]

    def extract(self, text: str):
        self.__text = text
        logging.info(f'Extracting Keywords')

        def flatten_nested_list(kw_list):
            x = []
            for sublist in kw_list:
                for item in sublist:
                    x.append(item.lower())
            return list(set(x))

        self.preprocess()

        with ThreadPoolExecutor() as pool:
            topic_rank = pool.submit(self.topic_rank_pke_extraction, self.__text)
            text_rank = pool.submit(self.text_rank_pke_extraction, self.__text)
            yake = pool.submit(self.yake_pke_extraction, self.__text)
            single_rank = pool.submit(self.single_rank_pke_extraction, self.__text)
            position_rank = pool.submit(self.position_rank_pke_extraction, self.__text)
            multipartite = pool.submit(self.multi_partite_rank_pke_extraction, self.__text)
            sg_scake = pool.submit(self.textacy_graph_extraction, self.__text)

            x = (topic_rank.result(), text_rank.result(), yake.result(),
                 single_rank.result(), position_rank.result(), multipartite.result(),
                 sg_scake.result())
            x = flatten_nested_list(x)
            return [list(item)[0] for item in textacy.extract.utils.aggregate_term_variants(set(x))]


class NamedEntitiesExtractor:
    def __init__(self, spacy_model):
        self.__spacy_model = spacy_model
        self.__entity_labels = [
            "DATE",
            "GPE",
            "LANGUAGE",
            "LOC",
            "NORP",
            "ORG",
            "PERSON",
            "PRODUCT",
        ]
        self.__entity_explain = "\n".join(
            f"{e} -> {spacy.explain(e)}" for e in self.__entity_labels
        )

    def explain_entity_labels(self) -> str:
        return self.__entity_explain

    def extract(self, text: str) -> List[Tuple[str, str]]:
        logging.info(f'Extracting Named Entities')
        return list(set([
            (ent.text.lower(), ent.label_)
            for ent in self.__spacy_model(text).ents
            if ent.label_ in self.__entity_labels
        ]))


class TextEnrichmentPipeline:
    def __init__(self,  spacy_model, stop_words: list):
        self.__text = None
        self.__stop_words = stop_words
        self.__spacy_model = spacy_model
        logging.info(f'\nLoading Entity Extractor')
        self.__entity_extractor = NamedEntitiesExtractor(spacy_model=self.__spacy_model)
        logging.info(f'\nLoading Statistics Extractor')
        self.__stats_extractor = StatisticsExtractor(stopwords=self.__stop_words)
        logging.info(f'\nLoading Keyword Extractor')
        self.__keyword_extractor = KeywordExtractor(stopwords=self.__stop_words, spacy_lang_model=self.__spacy_model)
        # logging.info(f'\nLoading Sentiment Extractor')
        # self.__sentiment_extractor = SentimentExtractor()

    def get_components(self):
        return {
            'entity_extractor': self.__entity_extractor,
            'statistics_extractor': self.__stats_extractor,
            'keyword_extractor': self.__keyword_extractor,
            # 'sentiment_extractor': self.__sentiment_extractor
        }

    def enrich(self, text: str):
        self.__text = text
        return {
            'entities': self.__entity_extractor.extract(self.__text),
            'statistics': self.__stats_extractor.extract(self.__text),
            'keywords': self.__keyword_extractor.extract(self.__text),
            #'document_sentiment': self.__sentiment_extractor.extract(self.__text)
        }