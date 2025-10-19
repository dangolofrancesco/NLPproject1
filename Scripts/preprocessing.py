# Sostituisci l'intera classe in 4_Scripts/preprocessing.py

import gensim
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.downloader import load

# Blocco di download NLTK corretto (da Soluzione 1)
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
    Una classe per pulire il testo e generare embedding a livello di frase
    usando un modello GloVe pre-addestrato.
    """
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        print(f"Caricamento del modello GloVe '{model_name}'...")
        self.model = load(model_name)
        self.vector_size = self.model.vector_size
        self.stop_words = set(stopwords.words('english'))
        print("Modello GloVe caricato con successo.")

    def _clean_text(self, text: str) -> list:
        """Pulisce e tokenizza un singolo testo."""
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        return tokens

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Genera l'embedding per un singolo testo.
        
        """
        # 1. Pulisci il testo per ottenere i token
        tokens = self._clean_text(text)
        
        # 2. Converti i token in vettori usando il modello GloVe
        vectors = [self.model[w] for w in tokens if w in self.model]
        
        # 3. Se non ci sono parole valide, restituisci un vettore di zeri
        if not vectors:
            return np.zeros(self.vector_size)
        
        # 4. Calcola la media dei vettori e restituiscila
        return np.mean(vectors, axis=0)

    def embed_sentences(self, sentences: list or object) -> np.ndarray:
        """
        Genera embedding per una lista o una Serie di pandas di frasi.
        """
        embeddings = [self.get_embedding(str(text)) for text in sentences]
        return np.stack(embeddings).astype(np.float32)