"""
LSTM-based model for emotion intensity, empathy, and polarity prediction.
Implements LSTM with PyTorch including embeddings, recurrent layers, and task-specific heads.
"""

import torch
import torch.nn as nn


class LSTMEmpathyNet(nn.Module):
    """
    LSTM-based model for predicting emotion intensity, empathy, and polarity.
    Uses pre-computed GloVe embeddings from the dataset.
    
    Architecture:
        Input: (batch_size, seq_len, embedding_dim)
        ↓
        LSTM Layers (with optional bidirectional processing)
        ↓
        Dropout
        ↓
        Task-specific Heads:
            - Intensity Head (regression)
            - Empathy Head (regression)
            - Polarity Head (classification)
    
    Components:
        - Embeddings: Pre-computed GloVe vectors (100-dim)
        - Recurrent Layers: LSTM (handles sequences and temporal dependencies)
        - Task-specific Heads: 3 separate output layers for different tasks
    """
    
    def __init__(self, config: dict):
        """
        Initialize LSTM model.
        
        Args:
            config (dict): Configuration dictionary containing:
                - input_dim (int): Embedding dimension (e.g., 100 for GloVe)
                - hidden_dim (int): LSTM hidden state size (e.g., 128)
                - num_layers (int): Number of stacked LSTM layers (default: 2)
                - dropout (float): Dropout rate for regularization (default: 0.3)
                - bidirectional (bool): Use bidirectional LSTM (default: False)
                - num_classes (int): Number of polarity classes (e.g., 4)
        """
        super(LSTMEmpathyNet, self).__init__()
        
        self.input_dim = config['input_dim']              # Embedding dimension (e.g., 100)
        self.hidden_dim = config['hidden_dim']            # LSTM hidden size (e.g., 128)
        self.num_layers = config.get('num_layers', 2)     # Number of LSTM layers
        self.dropout = config.get('dropout', 0.3)         # Dropout rate
        self.bidirectional = config.get('bidirectional', False)  # Bidirectional option
        self.num_classes = config['num_classes']          # Number of polarity classes
        
        # LSTM Recurrent Layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dropout=self.dropout if self.num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=self.bidirectional
        )
        
        # Calculate LSTM output dimension
        # If bidirectional, output is concatenation of forward and backward hidden states
        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    
        # Regression head for emotion intensity (continuous value)
        self.intensity_head = nn.Linear(lstm_output_dim, 1)
        
        # Regression head for empathy (continuous value)
        self.empathy_head = nn.Linear(lstm_output_dim, 1)
        
        # Classification head for emotional polarity (4 classes: 0, 1, 2, 3)
        self.polarity_head = nn.Linear(lstm_output_dim, self.num_classes)
    
    def forward(self, x, lengths=None):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, max_length, input_dim)
                       Contains pre-computed GloVe embeddings for each word in sequence
            lengths (Tensor, optional): Actual sequence lengths before padding
                                       Shape: (batch_size,)
                                       Used for efficient sequence packing
        
        Returns:
            dict: Dictionary containing predictions:
                - 'intensity': Emotion intensity predictions (batch_size, 1)
                - 'empathy': Empathy predictions (batch_size, 1)
                - 'polarity': Polarity logits (batch_size, num_classes)
        """
        batch_size = x.size(0)
        

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            sorted_lengths, sorted_idx = torch.sort(lengths_cpu, descending=True)
            x = x[sorted_idx]
            
            # Pack the sequences (removes padding)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            
            # Pass through LSTM
            # output: all hidden states at each time step
            # (hidden, cell): final hidden and cell states
            packed_output, (hidden, cell) = self.lstm(packed_input)
            
            # Unpack the sequences (restore original format with padding)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Unsort to restore original batch order
            _, unsorted_idx = torch.sort(sorted_idx)
            output = output[unsorted_idx]
            
            # Get the last relevant output for each sequence
            # Use lengths to index the last real output (before padding)
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, output.size(2))
            last_output = output.gather(1, idx).squeeze(1)
        else:
            # If no lengths provided, process normally
            output, (hidden, cell) = self.lstm(x)
            last_output = output[:, -1, :]  # Take last timestep
        
        features = self.dropout_layer(last_output)
        
        # Emotion intensity (regression)
        intensity = self.intensity_head(features)
        
        # Empathy (regression)
        empathy = self.empathy_head(features)
        
        # Emotional polarity (classification)
        polarity = self.polarity_head(features)
        
        return {
            'intensity': intensity,
            'empathy': empathy,
            'polarity': polarity
        }
    
    def get_model_info(self):
        """
        Returns a summary of the model configuration and parameters.
        
        Returns:
            dict: Model information including architecture details and parameter counts
        """
        return {
            'model_type': 'LSTM',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

if __name__ == "__main__":
    """
    Example of how to create and use the LSTM model
    """
    
    # Configuration
    config = {
        'input_dim': 100,           # GloVe embedding dimension
        'hidden_dim': 128,          # LSTM hidden size
        'num_layers': 2,            # Number of LSTM layers
        'dropout': 0.3,             # Dropout rate
        'bidirectional': True,      # Use bidirectional LSTM
        'num_classes': 4            # Number of polarity classes
    }
    
    # Create model
    model = LSTMEmpathyNet(config)
    
    # Print model information
    print("=" * 60)
    print("LSTM Model Information")
    print("=" * 60)
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)
    
    # Example forward pass
    batch_size = 8
    seq_len = 20
    embedding_dim = 100
    
    # Create dummy input (in practice, this comes from GloveSequenceEmbedder)
    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    dummy_lengths = torch.randint(10, seq_len, (batch_size,))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input, dummy_lengths)
    
    print(f"\nExample Forward Pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Intensity output shape: {outputs['intensity'].shape}")
    print(f"  Empathy output shape: {outputs['empathy'].shape}")
    print(f"  Polarity output shape: {outputs['polarity'].shape}")
    print("=" * 60)
