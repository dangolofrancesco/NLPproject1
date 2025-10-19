import torch
import pandas as pd
from torch.utils.data import Dataset
import math
import numpy as np

class EmpathyDataset(Dataset):
    """
    Dataset PyTorch che gestisce il caricamento, il preprocessing e la vettorizzazione
    dei dati per il task di empatia.
    """
    def __init__(self, csv_path: str, embedder):
        """
        Args:
            csv_path (str): Il percorso del file CSV.
            embedder: Un'istanza di una classe embedder (es. GloveEmbedder)
                      che ha un metodo .embed_sentences().
        """
        self.embedder = embedder
        
        # 1. Carica e pulisce il DataFrame
        df = self._load_and_clean_data(csv_path)
        
        # 2. Estrae le feature (testo -> embedding)
        print(f"Generating embeddings for {len(df)} samples...")
        self.features = self.embedder.embed_sentences(df["text"])
        
        # 3. Estrae le etichette
        self.intensity = df['Emotion'].values.astype(np.float32)
        self.empathy = df['Empathy'].values.astype(np.float32)
        # Convertiamo la polarità in una singola etichetta di classe (0, 1, 2, 3)
        self.polarity = df['EmotionalPolarity'].values.astype(int)
        
        # 4. Converte in tensori
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.intensity_tensor = torch.tensor(self.intensity, dtype=torch.float32).unsqueeze(1)
        self.empathy_tensor = torch.tensor(self.empathy, dtype=torch.float32).unsqueeze(1)
        self.polarity_tensor = torch.tensor(self.polarity, dtype=torch.long) # CrossEntropyLoss vuole Long

    def _load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Carica il CSV e applica la pulizia specifica del dataset."""
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        
        # Rimuove righe con valori mancanti nelle colonne chiave
        df.dropna(subset=['text', 'Emotion', 'Empathy', 'EmotionalPolarity'], inplace=True)
        
        # Corregge la colonna EmotionalPolarity come nel tuo codice originale
        df["EmotionalPolarity"] = df["EmotionalPolarity"].apply(
            lambda x: math.ceil(x) if isinstance(x, (int, float)) and str(x).endswith('.5') else x
        )
        # Assicura che i valori siano interi
        df = df[pd.to_numeric(df['EmotionalPolarity'], errors='coerce').notna()]
        df['EmotionalPolarity'] = df['EmotionalPolarity'].astype(int)

        # Se una classe '3' non esiste nel dev set, la aggiungiamo per coerenza di forma
        if 3 not in df['EmotionalPolarity'].unique():
            print("Warning: The '3' polarity class is not present in this dataset.")

        print(f"Loaded and cleaned data from {csv_path}. Number of samples:{len(df)}")
        return df

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, idx):
        return {
            'features': self.features_tensor[idx],
            'labels': {
                'intensity': self.intensity_tensor[idx],
                'empathy': self.empathy_tensor[idx],
                'polarity': self.polarity_tensor[idx]
            }
        }


class InferenceDataset(Dataset):
    """
    Dataset PyTorch per l'inferenza, gestisce solo ID e features.
    """
    def __init__(self, csv_path: str, embedder, id_column: str, text_column: str):
        """
        Args:
            csv_path (str): Percorso del file CSV di test.
            embedder: Un'istanza di GloveEmbedder già inizializzata.
            id_column (str): Nome della colonna contenente gli ID.
            text_column (str): Nome della colonna contenente il testo.
        """
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df.dropna(subset=[text_column], inplace=True)
        
        self.ids = df[id_column].values
        
        print(f"Generazione degli embedding per {len(df)} campioni di test...")
        features = embedder.embed_sentences(df[text_column])
        self.features_tensor = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'features': self.features_tensor[idx]
        }