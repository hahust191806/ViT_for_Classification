import math 
import torch 
from torch import nn 

import config

from model.activation.gelu import GELU
from model.layer.attention import MultiHeadAttention


class MLP(nn.Module):
    """
        A multi-layer perceptron module. 
        Input: x (batch_size, num_batches, hidden_size)
    """
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"]) 
        self.activation = GELU()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self, x): # Input: x (batch_size, num_batches, hidden_size)
        x = self.dense_1(x) # (batch_size, num_batches, intermediate_size)
        x = self.activation(x) # (batch_size, num_batches, intermediate_size)
        x = self.dense_2(x) # (batch_size, num_batches, hidden_size)
        x = self.dropout(x) # (batch_size, num_batches, hidden_size)
        return x # (batch_size, num_batches, hidden_size)
    

class Block(nn.Module):
    """
        A single transformer block. 
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config) # (batch_size, num_batches, hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config["hidden_size"])
        
    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x)) # (batch_size, num_batches, hidden_size)
        # Skip connection
        x = x + mlp_output # (batch_size, num_batches, hidden_size)
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs) # (batch_size, num_batches, hidden_size), (batch_size, num_batches, num_batches)
        
class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config) # (batch_size, num_batches, hidden_size), (batch_size, num_batches, num_batches)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False): # 
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions) # x: (batch_size, num_batches, hidden_size), attention_probs: (batch_size, num_batches, num_batches)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions) 