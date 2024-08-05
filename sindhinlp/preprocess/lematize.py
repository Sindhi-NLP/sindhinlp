import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

def load_models_and_tokenizers() -> tuple:
    """
    Loads the encoder and decoder models along with input and target tokenizers and configuration.

    This function loads the pre-trained encoder and decoder models from the specified file paths.
    It also loads the input and target tokenizers and the configuration settings required for the
    lemmatization process.

    Returns:
    tuple: A tuple containing the following elements:
        - encoder_model (tensorflow.keras.Model): The loaded encoder model.
        - decoder_model (tensorflow.keras.Model): The loaded decoder model.
        - input_tokenizer (Tokenizer): The loaded input tokenizer.
        - target_tokenizer (Tokenizer): The loaded target tokenizer.
        - config (dict): The loaded configuration settings.
    """
    encoder_model = load_model('sindhinlp/models/encoder_model.h5')
    decoder_model = load_model('sindhinlp/models/decoder_model.h5')

    with open('sindhinlp/models/input_tokenizer.pickle', 'rb') as handle:
        input_tokenizer = pickle.load(handle)

    with open('sindhinlp/models/target_tokenizer.pickle', 'rb') as handle:
        target_tokenizer = pickle.load(handle)

    with open('sindhinlp/models/config.pickle', 'rb') as handle:
        config = pickle.load(handle)

    return encoder_model, decoder_model, input_tokenizer, target_tokenizer, config


def lemmatize(input_text:str) -> str:
    """
    Lemmatizes the given Sindhi input text using a pre-trained sequence-to-sequence model.

    This function loads the necessary models and tokenizers, processes the input text,
    and generates the lemmatized version of the input using a pre-trained encoder-decoder model.

    Parameters:
    input_text (str): The Sindhi text to be lemmatized.

    Returns:
    str: The lemmatized version of the input text.
    """
    encoder_model, decoder_model, input_tokenizer, target_tokenizer, config = load_models_and_tokenizers()
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=config['max_input_len'], padding='post')

    states_value = encoder_model.predict(input_seq)

    
    target_seq = np.zeros((1, 1))
    
    target_seq[0, 0] = 2  
    
    stop_condition = False
    decoded_lemma = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word.get(sampled_token_index, '')
        decoded_lemma += sampled_char

        if (sampled_char == '' or len(decoded_lemma) > config['max_target_len']):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_lemma
