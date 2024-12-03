# models/intent_modules/model_lstm_int_bbox.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerIntBbox(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nhead, dropout, observe_length, max_len=100):
        super(TransformerIntBbox, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.observe_length = observe_length  # Adding observe_length as an attribute

        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)

        # Linear transformation to map input_dim to hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer for intent prediction
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Extract bounding box sequence
        bbox = data['bboxes'][:, :self.observe_length, :].type(torch.FloatTensor).to(data['bboxes'].device)
        
        # Map input_dim to hidden_dim
        x = self.input_fc(bbox)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Use only the output of the last time step for classification
        x = x[:, -1, :]

        # Output layer
        output = self.output_fc(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant "pe" matrix with values dependent on position and dimension
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
