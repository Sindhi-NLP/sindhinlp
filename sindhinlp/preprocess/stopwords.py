from typing import FrozenSet
import pandas as pd
def get_stopwords() -> FrozenSet:
    """
    Gives a set of stopwords from a CSV file.

    Parameters:
    No parameters
    
    Returns:
    set: set of stopwords.
    """
    stopwords = frozenset(pd.read_csv('sindhinlp/data/Stopwords.csv')['Stopwords'])
    
    return stopwords


def get_extend_stopwords(extended_set:set) -> FrozenSet[str]:
    """
    Gives extended a set of stopwords with additional stopword.

    Parameters:
    set: set of stopwords to be added in already existing stopwords from this library
    
    Returns:
    set: extended set of stopwords with union of new and existing stopwords.
    """
    stopwords = frozenset(pd.read_csv('sindhinlp/data/Stopwords.csv')['Stopwords'])
    new_stopwpords = extended_set.union(stopwords)
    
    return frozenset(new_stopwpords)