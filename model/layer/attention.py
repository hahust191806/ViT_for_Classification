import torch 
from torch import nn 
import math 


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.
    attention_head_size = hidden_size // num_attention_head: dimension of query, key, value
    input x: (batch_size, sequence_length, attention_head_size)
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # create the query, key and value projection layers 
        self.query = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        self.key = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        self.value = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # x: (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x) # -> (batch_size, sequence_length, attention_head_size)
        key = self.key(x) # -> (batch_size, sequence_length, attention_head_size)
        value = self.value(x) # -> (batch_size, sequence_length, attention_head_size)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs) # -> (batch_size, sequence_length, attention_head_size)
    

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"] # config hidden_size 
        self.num_attention_heads = config["num_attention_heads"] # config number of attention heads 
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads 
        self.all_head_size = self.num_attention_heads * self.attention_head_size # == self.hidden_size 
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"] 
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            ) # -> (batch_size, sequence_length, attention_head_size) : sequence_length == num_patches 
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads] # -> (batch_size, sequence_length, attention_head_size)
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1) # -> (batch_size, sequence_length, hidden_size)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output) # -> (batch_size, sequence_length, hidden_size)
        attention_output = self.output_dropout(attention_output) # -> (batch_size, sequence_length, hidden_size)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs) # -> (batch_size, sequence_length, hidden_size)