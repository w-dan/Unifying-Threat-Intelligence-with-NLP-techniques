from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

def preprocess_corpus(text: str) -> str:
    """
    Preprocesses a given text by removing stopwords and setting lowercase.

    Args:
        text: The text to be preprocessed.

    Returns:
        The preprocessed text.
    """

    nltk.download('stopwords')
    nltk.download('punkt')

    tokens: List[str] = word_tokenize(text.lower())
    stop_words: set[str] = set(stopwords.words('english'))
    filtered_tokens: List[str] = [token for token in tokens if token not in stop_words]

    preprocessed_text: str = ' '.join(filtered_tokens)

    return preprocessed_text