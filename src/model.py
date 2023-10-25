import math

import torch
import torch.nn as nn


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Arguments:
#             x: torch.Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


class SwipeCurveEncoderTransformer(nn.Module):
    """
    Transformer-based Curve encoder takes in a sequence of vectors and creates a representation
    of a swipe gesture on a samrtphone keyboard.
    Each vector contains information about finger trajectory at a time step.
    It contains:
    * x coordinate
    * y coordinate
    * Optionally: dx/dt
    * Optionally: dy/dt
    * Optionally: keyboard key that has x and y coordinates within its boundaries
    """

    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)