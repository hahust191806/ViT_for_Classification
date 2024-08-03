import torch 
from torch import nn 


class PatchEmbeddings(nn.Module):
    """
        Convert the image into patches and then project them into a vector space
        Input size: (batch_size, num_channels, image_size, image_size)
        Output size: (batch_size, num_patches, hidden_size)
    """ 
    
    def __init__(self, config): # -> config: module consist hyperparameters 
        super().__init__()
        self.image_size = config['image_size'] # config image_size of input 
        self.patch_size = config['patch_size'] # config patch_size of each patch 
        self.num_channels = config['num_channels'] # config num_channels of image input 
        self.hidden_size = config['hidden_size'] # config hidden_size 
        # calcalate the number of patches from the image size and patch size 
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # create a projection layer to convert the image into patches 
        # the layer projects each patch into a vector of size hidden_size 
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel__size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x): # input x: (batch_size, num_channels, image_size, image_size)
        x = self.projection(x) # -> (batch_size, hidden_size, patch_size, patch_size)
        x = x.flatten(2).transpose(1, 2)# -> (batch_size, hidden_size, sqrt(num_patches) * sqrt(num_patches)) -> (batch_size, num_patches, hidden_size) 
        return x # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)dddddddddddddddddddđddddđ
    
class Embeddings(nn.Module):
    """
        Combine the patch embeddings with the class token and position embeddings. 
    """ 
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, num_patches + 1, hidden_size)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x  # (batch_size, num_patches + 1, hidden_size)