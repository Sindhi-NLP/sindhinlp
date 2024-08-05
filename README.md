# Sindhi NLP

A package for Sindhi language processing including stopword removal, POS tagging, and lemmatization.

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/sindhinlp.git
cd sindhinlp
```

### Install dependencies

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

### Install the package

Install the package in editable mode:

```bash
pip install -e .
```

## Usage

### Tokenization

#### Tokenize Words

```python
from sindhinlp.preprocess.tokenize import tokenize_words

text = "سنڌي زبان جو مونھنجو علم، تاريخ."
word_tokens = tokenize_words(text)
print(word_tokens)
```

#### Tokenize Sentences

```python
from sindhinlp.preprocess.tokenize import tokenize_sentences

text = "سنڌي زبان جو مونھنجو علم، تاريخ ۽ ثقافت ۾ سنڌ جي ڊوڪريون، لوڪيون ۽ ڪتابون ۾ سمجهايو جيون ڪرتوت۔ سنڌي جي عوام کي سمجهاڻ جو رويا پنهنجي زبان ۾ سائنس، فنون ۽ ساينس ڪان متعلق احاطو سوانح ۽ ڪرتوت ۽ محاورن جو معلومات ديندو آهي."
sentence_tokens = tokenize_sentences(text)
print(sentence_tokens)
```

#### Custom Tokenize

```python
from sindhinlp.preprocess.tokenize import custom_tokenize

text = "سنڌي زبان، جي مونھنجو علم، تاريخ ۽ ثقافت"
custom_tokens = custom_tokenize(text, "،")
print(custom_tokens)
```

### Stopwords

#### Get Stopwords

```python
from sindhinlp.preprocess.stopwords import get_stopwords

stopwords = get_stopwords()
print(stopwords)
```

#### Get Extended Stopwords

```python
from sindhinlp.preprocess.stopwords import get_extend_stopwords

extended_set = {'نئون', 'اضافي'}
extended_stopwords = get_extend_stopwords(extended_set)
print(extended_stopwords)
```

### Text Preprocessing

#### Remove Special Characters

```python
from sindhinlp.preprocess.text_preprocessing import remove_special_characters

text = "سنڌي زبان جو مونھنجو علم، تاريخ ۽ ثقافت۔ ڪتاب جو عنوان: 'سنڌ جو آسمان'"
clean_text = remove_special_characters(text)
print(clean_text)
```

#### Remove Stopwords

```python
from sindhinlp.preprocess.text_preprocessing import remove_stopwords

text = "محاورن جو معلومات ديندو آهي"
stopword_removed_text = remove_stopwords(text)
print(stopword_removed_text)
```

#### Remove Numbers

```python
from sindhinlp.preprocess.text_preprocessing import remove_number

text = "هي 1234 ٽيسٽ 5678 مواد آهي 90 نمبر جي وچ ۾."
result = remove_number(text)
print(result)
```

#### Remove URLs

```python
from sindhinlp.preprocess.text_preprocessing import remove_urls

text = "ھي لنڪ چيڪ ڪريو: http://example.com ۽ پڻ وزٽ ڪريو https://www.example.org وڌيڪ معلومات لاءِ."
result = remove_urls(text)
print(result)
```

#### Remove Emojis

```python
from sindhinlp.preprocess.text_preprocessing import remove_emoji

text = "ھي ٽيسٽ مواد آھي 😊🚀🌟 جي وچ ۾."
result = remove_emoji(text)
print(result)
```

### Lemmatization

```python
from sindhinlp.preprocess.lematize import lemmatize

word = "رهندو"
lemma = lemmatize(word)
print(lemma)
```

### POS Tagging

```python
from sindhinlp.models.pos_tagging import pos_tags

sentence = "موسم بهترين آهي."
tags = pos_tags(sentence)
print(tags)
```

### Load Dataset

```python
from sindhinlp.data.load_data import load_dataset

# Load default dataset
df = load_dataset()
print(df.head())

# Load specific dataset
df = load_dataset('article_dataset')
print(df.head())
```

## Running Tests

To run tests, use the following command:

```bash
python -m unittest discover -s test
```

## Contributing

We welcome contributions to this project! To contribute, please follow the steps mentioned in [CONTRIBUTING](CONTRIBUTING.md)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
