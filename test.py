import unittest
import pandas as pd
from sindhinlp.preprocess.tokenize import tokenize_words,tokenize_sentences, costum_tokenize
from sindhinlp.preprocess.stopwords import get_stopwords, get_extend_stopwords
from sindhinlp.preprocess.text_preprocessing import remove_special_characters, remove_stopwords, remove_urls, remove_emoji,remove_number
from sindhinlp.preprocess.lematize import lemmatize
from sindhinlp.models.pos_tagging import pos_tags
from sindhinlp.data.load_data import load_dataset
from typing import FrozenSet

class TestTokenization(unittest.TestCase):

    def test_tokenize_words(self):
        text = "سنڌي زبان جو مونھنجو علم، تاريخ."
        expected_tokens = ['سنڌي', 'زبان', 'جو', 'مونھنجو', 'علم', 'تاريخ']
        tokens = tokenize_words(text)
        self.assertEqual(tokens, expected_tokens)

    def test_tokenize_sentences(self):
        text = "سنڌي زبان جو مونھنجو علم، تاريخ ۽ ثقافت ۾ سنڌ جي ڊوڪريون، لوڪيون ۽ ڪتابون ۾ سمجهايو جيون ڪرتوت۔ سنڌي جي عوام کي سمجهاڻ جو رويا پنهنجي زبان ۾ سائنس، فنون ۽ ساينس ڪان متعلق احاطو سوانح ۽ ڪرتوت ۽ محاورن جو معلومات ديندو آهي."
        expected_sentences = [
            'سنڌي زبان جو مونھنجو علم، تاريخ ۽ ثقافت ۾ سنڌ جي ڊوڪريون، لوڪيون ۽ ڪتابون ۾ سمجهايو جيون ڪرتوت۔',
            'سنڌي جي عوام کي سمجهاڻ جو رويا پنهنجي زبان ۾ سائنس، فنون ۽ ساينس ڪان متعلق احاطو سوانح ۽ ڪرتوت ۽ محاورن جو معلومات ديندو آهي.'
        ]
        sentences = tokenize_sentences(text)
        self.assertEqual(sentences, expected_sentences)

    def test_custom_tokenize(self):
        text = "سنڌي زبان، جي مونھنجو علم، تاريخ ۽ ثقافت"        
        expected_sentences = ['سنڌي زبان', ' جي مونھنجو علم', ' تاريخ ۽ ثقافت']
        sentences = costum_tokenize(text,"،")
        self.assertEqual(sentences , expected_sentences)

    def test_custom_tokenize(self):
        text = "سنڌي زبان، جي مونھنجو علم، تاريخ ۽ ثقافت"        
        expected_sentences = ['سنڌي زبان', ' جي مونھنجو علم', ' تاريخ ۽ ثقافت']
        sentences = costum_tokenize(text,"،")
        self.assertEqual(sentences , expected_sentences)

    def test_get_stopwords(self):
        stopwords = get_stopwords()
        assert isinstance(stopwords, FrozenSet)
        assert len(stopwords) == 606
        assert 'هڏا' in stopwords
        assert 'ڇا' in stopwords 

    def test_get_extend_stopwords(self):
        
        extended_set = {'نئون', 'اضافي'}
        extended_stopwords = get_extend_stopwords(extended_set)
        assert isinstance(extended_stopwords, FrozenSet)
        assert len(extended_stopwords) == 608  # 606 from original + 2 new
        assert 'هڏا' in extended_stopwords
        assert 'ڇا' in extended_stopwords 
        assert 'نئون' in extended_stopwords
        assert 'اضافي' in extended_stopwords

    def test_get_extend_stopwords_empty_set(self):
        extended_stopwords = get_extend_stopwords(set())
        assert isinstance(extended_stopwords, FrozenSet)
        assert len(extended_stopwords) == 606 # Same as original set
        assert extended_stopwords == get_stopwords()

    def test_get_extend_stopwords_duplicate(self):
        extended_set = {'هڏا', 'نئون'}  # 'هڏا' is already in the original set
        extended_stopwords = get_extend_stopwords(extended_set)
        assert len(extended_stopwords) == 607  # 606 from original + 1 new
        assert isinstance(extended_stopwords, FrozenSet)

    def test_remove_special_characters(self):
        text = "سنڌي زبان جو مونھنجو علم، تاريخ ۽ ثقافت۔ ڪتاب جو عنوان: 'سنڌ جو آسمان'"
        expected_text = "سنڌي زبان جو مونھنجو علم تاريخ ۽ ثقافت ڪتاب جو عنوان سنڌ جو آسمان"
        clean_text = remove_special_characters(text)
        self.assertEqual(clean_text, expected_text)

    def test_remove_stopwords(self):
        text = "محاورن جو معلومات ديندو آهي"
        expected_text = "محاورن معلومات ديندو"
        stopword_removed_text = remove_stopwords(text)
        self.assertEqual(stopword_removed_text , expected_text)

    def test_remove_number(self):
        sindhi_text_with_numbers = "هي 1234 ٽيسٽ 5678 مواد آهي 90 نمبر جي وچ ۾."
        expected_sindhi_result_no_numbers = "هي ٽيسٽ مواد آهي نمبر جي وچ ۾."
        result_sindhi_no_numbers = remove_number(sindhi_text_with_numbers)
        assert result_sindhi_no_numbers == expected_sindhi_result_no_numbers, f"Expected: '{expected_sindhi_result_no_numbers}', but got: '{result_sindhi_no_numbers}'"

    def test_remove_urls(self):
        sindhi_text_with_urls = "ھي لنڪ چيڪ ڪريو: http://example.com ۽ پڻ وزٽ ڪريو https://www.example.org وڌيڪ معلومات لاءِ."
        expected_sindhi_result_no_urls = "ھي لنڪ چيڪ ڪريو: ۽ پڻ وزٽ ڪريو وڌيڪ معلومات لاءِ."
        result_sindhi_no_urls = remove_urls(sindhi_text_with_urls)
        assert result_sindhi_no_urls == expected_sindhi_result_no_urls, f"Expected: '{expected_sindhi_result_no_urls}', but got: '{result_sindhi_no_urls}'"
    
    def test_remove_emoji(self):
        sindhi_text_with_emojis = "ھي ٽيسٽ مواد آھي 😊🚀🌟 جي وچ ۾."
        expected_sindhi_result_no_emojis = "ھي ٽيسٽ مواد آھي جي وچ ۾."
        result_sindhi_no_emojis = remove_emoji(sindhi_text_with_emojis)
        assert result_sindhi_no_emojis == expected_sindhi_result_no_emojis, f"Expected: '{expected_sindhi_result_no_emojis}', but got: '{result_sindhi_no_emojis}'"

    def test_pos_tagger(self):
        sentence = ". موسم بهترين آهي."
        tags = pos_tags(sentence)
        assert tags == ['PERIOD', 'NOUN', 'ADJ', 'NOUN']

    def test_lemmatize(self):
        word = "رهندو"
        lemma = 'ره'
        assert lemmatize(word) == lemma

    def test_load_dataset_default(self):
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Unnamed: 0', 'article', 'link', 'title', 'genre']

    def test_load_dataset_article_dataset(mock_csv):
        df = load_dataset('article_dataset')
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Unnamed: 0', 'article', 'link', 'date', 'author']


if __name__ == '__main__':
    unittest.main()