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
    
        df = self._load_and_clean_data(csv_path)
        
        print(f"Generating embeddings for {len(df)} samples...")
        self.features = self.embedder.embed_sentences(df["text"])
        
        self.intensity = df['Emotion'].values.astype(np.float32)
        self.empathy = df['Empathy'].values.astype(np.float32)
        
        # Convertiamo la polarità in una singola etichetta di classe (0, 1, 2, 3)
        self.polarity = df['EmotionalPolarity'].values.astype(int)
        
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.intensity_tensor = torch.tensor(self.intensity, dtype=torch.float32).unsqueeze(1)
        self.empathy_tensor = torch.tensor(self.empathy, dtype=torch.float32).unsqueeze(1)
        self.polarity_tensor = torch.tensor(self.polarity, dtype=torch.long) # CrossEntropyLoss vuole Long

    def _load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Carica il CSV e applica la pulizia specifica del dataset."""
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        
        df.dropna(subset=['text', 'Emotion', 'Empathy', 'EmotionalPolarity'], inplace=True)
        
        df["EmotionalPolarity"] = df["EmotionalPolarity"].apply(
            lambda x: math.ceil(x) if isinstance(x, (int, float)) and str(x).endswith('.5') else x
        )
        
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
    PyTorch Dataset for inference, handles only IDs and features.
    """
    def __init__(self, csv_path: str, embedder, id_column: str, text_column: str):
        """
        Args:
            csv_path (str): Path to the test CSV file.
            embedder: An instance of GloveEmbedder already initialized.
            id_column (str): Name of the column containing IDs.
            text_column (str): Name of the column containing text.
        """
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df.dropna(subset=[text_column], inplace=True)
        
        self.ids = df[id_column].values
        
        print(f"Generating embeddings for {len(df)} test samples...")
        features = embedder.embed_sentences(df[text_column])
        self.features_tensor = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'features': self.features_tensor[idx]
        }


class RNNEmpathyDataset(Dataset):
    """
    PyTorch Dataset for RNN/LSTM/GRU models that handles loading, preprocessing,
    and sequence vectorization for the empathy task.
    """
    def __init__(self, csv_path: str, sequence_embedder):
        """
        Args:
            csv_path (str): Path to the CSV file.
            sequence_embedder: An instance of GloveSequenceEmbedder
                              that has an .embed_sentences() method.
        """
        self.sequence_embedder = sequence_embedder

        df = self._load_and_clean_data(csv_path)
   
        print(f"Generating sequence embeddings for {len(df)} samples...")
        self.features, self.lengths = self.sequence_embedder.embed_sentences(df["text"])
        
        self.intensity = df['Emotion'].values.astype(np.float32)
        self.empathy = df['Empathy'].values.astype(np.float32)
        
        # Convert polarity to a single class label (0, 1, 2, 3)
        self.polarity = df['EmotionalPolarity'].values.astype(int)
        
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.lengths_tensor = torch.tensor(self.lengths, dtype=torch.long)
        self.intensity_tensor = torch.tensor(self.intensity, dtype=torch.float32).unsqueeze(1)
        self.empathy_tensor = torch.tensor(self.empathy, dtype=torch.float32).unsqueeze(1)
        self.polarity_tensor = torch.tensor(self.polarity, dtype=torch.long)

    def _load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Loads the CSV and applies dataset-specific cleaning."""
        df = pd.read_csv(csv_path, on_bad_lines='skip')

        df.dropna(subset=['text', 'Emotion', 'Empathy', 'EmotionalPolarity'], inplace=True)
        
        df["EmotionalPolarity"] = df["EmotionalPolarity"].apply(
            lambda x: math.ceil(x) if isinstance(x, (int, float)) and str(x).endswith('.5') else x
        )
    
        df = df[pd.to_numeric(df['EmotionalPolarity'], errors='coerce').notna()]
        df['EmotionalPolarity'] = df['EmotionalPolarity'].astype(int)

        if 3 not in df['EmotionalPolarity'].unique():
            print("Warning: The '3' polarity class is not present in this dataset.")

        print(f"Loaded and cleaned data from {csv_path}. Number of samples: {len(df)}")
        return df

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, idx):
        return {
            'features': self.features_tensor[idx],      # Shape: (max_length, embedding_dim)
            'length': self.lengths_tensor[idx],         # Actual sequence length
            'labels': {
                'intensity': self.intensity_tensor[idx],
                'empathy': self.empathy_tensor[idx],
                'polarity': self.polarity_tensor[idx]
            }
        }


class RNNInferenceDataset(Dataset):
    """
    PyTorch Dataset for RNN inference, handles only IDs, sequence features, and lengths.
    """
    def __init__(self, csv_path: str, sequence_embedder, id_column: str, text_column: str):
        """
        Args:
            csv_path (str): Path to the test CSV file.
            sequence_embedder: An instance of GloveSequenceEmbedder already initialized.
            id_column (str): Name of the column containing IDs.
            text_column (str): Name of the column containing text.
        """
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df.dropna(subset=[text_column], inplace=True)
        
        self.ids = df[id_column].values
        
        print(f"Generating sequence embeddings for {len(df)} test samples...")
        features, lengths = sequence_embedder.embed_sentences(df[text_column])
        self.features_tensor = torch.tensor(features, dtype=torch.float32)
        self.lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'features': self.features_tensor[idx],  # Shape: (max_length, embedding_dim)
            'length': self.lengths_tensor[idx]      # Actual sequence length
        }