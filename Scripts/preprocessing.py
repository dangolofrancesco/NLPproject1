import gensim
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.downloader import load

# Blocco di download NLTK corretto 
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Una o più risorse NLTK non trovate. Le scarico ora...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

class GloveEmbedder:
    """
    A class to clean text and generate sentence-level embeddings
    using a pre-trained GloVe model.
    This version AVERAGES word vectors - suitable for ANNs.
    """
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        print(f"Loading GloVe model '{model_name}'...")
        self.model = load(model_name)
        self.vector_size = self.model.vector_size
        self.stop_words = set(stopwords.words('english'))
        print("GloVe model loaded successfully.")

    def _clean_text(self, text: str) -> list:
        """Cleans and tokenizes a single text."""
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        return tokens

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates the embedding for a single text by averaging word vectors.
        Returns a single vector of shape (vector_size,).
        """
        tokens = self._clean_text(text)
        
        # Convert tokens to vectors using the GloVe model
        vectors = [self.model[w] for w in tokens if w in self.model]
        
        # If there are no valid words, return a zero vector
        if not vectors:
            return np.zeros(self.vector_size)
        
        # Calculate and return the mean of the vectors
        return np.mean(vectors, axis=0)

    def embed_sentences(self, sentences: list or object) -> np.ndarray:
        """
        Generates embeddings for a list or pandas Series of sentences.
        Returns shape (num_sentences, vector_size).
        """
        embeddings = [self.get_embedding(str(text)) for text in sentences]
        return np.stack(embeddings).astype(np.float32)


class GloveSequenceEmbedder:
    """
    A class to clean text and generate sequence-level embeddings
    using a pre-trained GloVe model.
    This version PRESERVES word sequences - suitable for RNNs/LSTMs/GRUs.
    """
    def __init__(self, model_name="glove-wiki-gigaword-100", max_length=50):
        print(f"Loading GloVe model '{model_name}' for sequence embedding...")
        self.model = load(model_name)
        self.vector_size = self.model.vector_size
        self.max_length = max_length
        self.stop_words = set(stopwords.words('english'))
        print(f"GloVe sequence model loaded successfully. Max length: {max_length}")

    def _clean_text(self, text: str) -> list:
        """Cleans and tokenizes a single text."""
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        return tokens

    def get_sequence_embedding(self, text: str) -> tuple:
        """
        Generates a sequence of word embeddings for a single text.
        
        Returns:
            embeddings: np.ndarray of shape (max_length, vector_size) - padded sequence
            length: int - actual number of valid tokens (before padding)
        """
        tokens = self._clean_text(text)
        
        vectors = [self.model[w] for w in tokens if w in self.model]
        
        # Get actual sequence length (before padding)
        actual_length = len(vectors)
        
        # Truncate if too long
        if actual_length > self.max_length:
            vectors = vectors[:self.max_length]
            actual_length = self.max_length
        
        # Pad if too short
        if actual_length == 0:
            # If no valid words, return zero sequence
            vectors = [np.zeros(self.vector_size)]
            actual_length = 1
        
        # Create padded sequence
        padded_sequence = np.zeros((self.max_length, self.vector_size), dtype=np.float32)
        padded_sequence[:len(vectors)] = vectors
        
        return padded_sequence, actual_length

    def embed_sentences(self, sentences: list or object) -> tuple:
        """
        Generates sequence embeddings for a list or pandas Series of sentences.
        
        Returns:
            embeddings: np.ndarray of shape (num_sentences, max_length, vector_size)
            lengths: np.ndarray of shape (num_sentences,) - actual lengths before padding
        """
        embeddings_list = []
        lengths_list = []
        
        for text in sentences:
            embedding, length = self.get_sequence_embedding(str(text))
            embeddings_list.append(embedding)
            lengths_list.append(length)
        
        embeddings = np.stack(embeddings_list).astype(np.float32)
        lengths = np.array(lengths_list, dtype=np.int32)
        
        return embeddings, lengths