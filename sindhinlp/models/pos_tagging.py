import pycrfsuite

def convert_features(X:list) -> list:
    """
        Convert features to the format required by pycrfsuite.

        Parameters:
        X (list of dict): List of feature dictionaries.

        Returns:
        list of dict: List of feature dictionaries with string values.
    """
    
    return [{k: str(v) for k, v in x.items()} for x in X]


def prepare_sentence_for_tagging(sentence:str) -> list:
    """
        Prepare the sentence for tagging by generating feature dictionaries for each 
        token.

        Parameters:
        sentence (str): The input sentence to prepare for tagging.

        Returns:
        list of dict: List of feature dictionaries for each token in the sentence.
    """
    tokens = sentence.split()
    prepared_sentence = []
    
    for i, token in enumerate(tokens):
        token_dict = {'word': token}
        
        if i == 0:
            token_dict['BOS'] = 'True'
        else:
            token_dict['-1:word'] = tokens[i-1]
        
        if i == len(tokens) - 1:
            token_dict['EOS'] = 'True'
        else:
            token_dict['+1:word'] = tokens[i+1]
        
        prepared_sentence.append(token_dict)
    
    return prepared_sentence

def pos_tags(sentence:str) -> list:
    """
        Predict POS tags for the given sentence.

        Parameters:
        sentence (str): The input sentence to tag.

        Returns:
        list of str: List of predicted POS tags for each token in the sentence.
    """
    tagger = pycrfsuite.Tagger()
    
    tagger.open('sindhinlp\models\sindhiposmodel.crfsuite')
    
    prepared = prepare_sentence_for_tagging(sentence)
    
    tags = tagger.tag(convert_features(prepared))
    
    return tags