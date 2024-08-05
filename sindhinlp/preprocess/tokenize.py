import re
from typing import List

def tokenize_words(text : str) -> List[str]:
    """
    Tokenizes the given Sindhi text into words.

    Parameters:
    text (str): Input Sindhi text to tokenize.

    Returns:
    list: List of tokens (words) extracted from the text.
    """
    try: 
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        
        # Split text into words using whitespace characters as delimiters
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    except ValueError as e:
        raise ValueError(str(e))

def tokenize_sentences(text : str) -> List[str]:
    """
    Tokenizes the given Sindhi text into sentences.

    Parameters:
    text (str): Input Sindhi text to tokenize.

    Returns:
    list: List of tokens (sentences) extracted from the text.
    """
    try:
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        
        # Split text into sentences using punctuation marks as delimiters
        sentences = re.split(r'(?<=[।!.۔?])+ ', text)
        return sentences
    
    except ValueError as e:
        raise ValueError(str(e))


def costum_tokenize(text : str, delimiter : str) -> List[str]:
    """
    Tokenizes the given Sindhi text into sentences using custom delimiters.

    Parameters:
    text (str): Input Sindhi text to tokenize.
    delimiter (str): Delimiter to tokenize text. 
    
    Returns:
    list: List of tokens (sentences) extracted from the text.
    """
    
    try:
        if not isinstance(text , str):
            raise ValueError("Input Text must be String")
        
        # Split text into sentences using punctuation marks as delimiters
        sentences = text.split(delimiter)
        return sentences
    
    except ValueError as e:
        raise ValueError(str(e))