import math 
import torch 
from torch import nn 

from embedding.patch_embedding import Embeddings
from layer.encoder import Encoder


class ViTForClassfication(nn.Module):
    """
        The ViT for classificaiton
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # create the embedding module 
        self.embedding = Embeddings(config) # (batch_size, num_patches + 1, hidden_size)
        # create the transformer encoder module 
        self.encoder = Encoder(config) # x: (batch_size, num_batches, hidden_size)
        # create a linear layer to project the encoder's output to the number of classes 
        self.classifier = nn.Linear(self.hidden_size, self.num_classes) # (batch_size, num_batches + 1, num_classes)
        # initialize the weights 
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False): # x: (batch_size, num_channels, image_size, image_size)
        # calculate the embedding output 
        embedding_output = self.embedding(x) # (batch_size, num_patches + 1, hidden_size)
        # calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions) # encoder_output: (batch_size, num_batches, hidden_size), all_attentions: (batch_size, num_batches, num_batches)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :]) # (1, num_classes)
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)