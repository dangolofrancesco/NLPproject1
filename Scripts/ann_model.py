import torch
import torch.nn as nn
from typing import List, Dict

class DeepEmpathyNet(nn.Module):
    """
    A multi-head deep neural network for empathy, intensity, and polarity prediction.

    This model consists of a shared backbone of dense layers followed by three
    separate linear heads for each specific task.
    """
    def __init__(self, config: Dict):
        """
        Initializes the DeepEmpathyNet model using a configuration dictionary.

        Args:
            config (Dict): A dictionary containing model configuration parameters.
                Expected keys:
                - 'input_dim' (int): Dimensionality of the input features.
                - 'hidden_dims' (List[int]): List of hidden layer sizes.
                - 'dropout' (float): Dropout rate.
                - 'activation' (str): Name of the activation function.
                - 'norm_type' (str): 'layernorm' or 'batchnorm'.
                - 'num_classes' (int): Number of classes for the polarity head.
        """
        super().__init__()
        self.config = config
        
        self._build_backbone()
        self._build_heads()

    def _build_backbone(self):
        """Builds the shared layers (backbone) of the network."""
        hidden_dims = self.config.get("hidden_dims", [512, 256, 128])
        dropout_rate = self.config.get("dropout", 0.3)
        activation_name = self.config.get("activation", "relu").lower()
        norm_type = self.config.get("norm_type", "layernorm").lower()

        # Activation functions
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leakyrelu": nn.LeakyReLU(0.01),
        }
        act_func = activations.get(activation_name, nn.ReLU())

        # Normalization layer type
        def get_norm(h_dim):
            if norm_type == "batchnorm":
                return nn.BatchNorm1d(h_dim)
            return nn.LayerNorm(h_dim)

        self.backbone = nn.ModuleList()
        prev_dim = self.config["input_dim"]

        for h_dim in hidden_dims:
            self.backbone.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h_dim),
                    get_norm(h_dim),
                    act_func,
                    nn.Dropout(dropout_rate),
                )
            )
            prev_dim = h_dim
        
        # Store the dimension of the last hidden layer for the heads
        self.last_hidden_dim = prev_dim

    def _build_heads(self):
        """Builds the output heads for each task."""
        self.intensity_head = nn.Linear(self.last_hidden_dim, 1)
        self.empathy_head = nn.Linear(self.last_hidden_dim, 1)
        self.polarity_head = nn.Linear(self.last_hidden_dim, self.config.get("num_classes", 4))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the outputs from each head,
                                     keyed by 'intensity', 'empathy', and 'polarity'.
        """
        # Pass input through the shared backbone
        for layer in self.backbone:
            x = layer(x)

        # Calculate outputs for each head from the final backbone representation
        outputs = {
            "intensity": self.intensity_head(x),
            "empathy": self.empathy_head(x),
            "polarity": self.polarity_head(x),
        }
        return outputs