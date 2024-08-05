import re
from .tokenize import tokenize_words
from .stopwords import get_stopwords
def remove_special_characters(text: str) -> str:
    """
    Removes all special characters from the input Sindhi string.

    Parameters:
    text (str): Input Sindhi string containing special characters.

    Returns:
    str: String with special characters removed.
    """
    try: 
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        pattern = r'[^\u0621-\u06BE \u06FD\s]' 
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text
    except ValueError as e:
        raise ValueError(str(e))


def remove_stopwords(text:str) -> str:
    """
    Removes stopwords from the input Sindhi string.

    Parameters:
    text (str): Input Sindhi string containing stopwords.

    Returns:
    str: String with stopwords removed.
    """
    try:
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        words = tokenize_words(text)
        stopwords = get_stopwords()
        return " ".join([word for word in words if word not in stopwords])
    except TypeError as e:
        raise ValueError(str(e))
    
def remove_number(text: str) -> str:
    """
    Removes numbers from the input string and ensures no extra spaces are left.

    Parameters:
    text (str): Input string containing numbers.

    Returns:
    str: String with numbers and extra spaces removed.
    """
    try:
        if not isinstance(text, str):
            raise ValueError("Input Text must be String")
        text_without_numbers = re.sub(r'\d+', '', text)
        text_without_extra_spaces = re.sub(r'\s+', ' ', text_without_numbers).strip()
        return text_without_extra_spaces
    except TypeError as e:
        raise ValueError(str(e))

def remove_urls(text:str) -> str:
    """
    Removes URLs from the input string.

    Parameters:
    text (str): Input string containing URLs.

    Returns:
    str: String with URLs removed.
    """
    try:
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        url_pattern = r'http[s]?://\S+|www\.\S+'
        text_without_urls = re.sub(url_pattern, '', text)
        text_without_extra_spaces = re.sub(r'\s+', ' ', text_without_urls).strip()
        return text_without_extra_spaces
    except TypeError as e:
        raise ValueError(str(e))

def remove_emoji(text:str) -> str:
    """
    Removes emojis from the input string.

    Parameters:
    text (str): Input string containing emojis.

    Returns:
    str: String with emojis removed.
    """
    try:
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        text_without_emojis = emoji_pattern.sub(r'', text)
        text_without_extra_spaces = re.sub(r'\s+', ' ', text_without_emojis).strip()
        return text_without_extra_spaces

    except TypeError as e:
        raise ValueError(str(e))