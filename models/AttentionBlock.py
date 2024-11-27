import torch.nn as nn
import torch
from torch.nn import functional as F
from MultiHeadAttention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()

        # The input to the network is [B, T, C]
        self.network=nn.Sequential(
            # 1) Increasing the dimension to [B, T, 4C]
            nn.Linear(embed_dim, 4*embed_dim),
            #
            # 2) ReLU
            nn.ReLU(),
            #
            # 3) decreasing the dimension back to [B, T, C]
            nn.Linear(4*embed_dim, embed_dim),
            #
            # 4) Adding the dropout
            nn.Dropout(dropout),
        )
    
    def forward(self, scores):
        return self.network(scores)

class AttentionBlock(nn.Module):
    def __init__(self,embed_dim,n_heads):
        super().__init__()

        # 1) Instantiating the MHA 
        # Input : [B, T, C]
        # Output: [B, T, C]
        self.mha = MultiHeadAttention(embed_dim,n_heads)

        # 2) Creating the layer normalization module 1
        # Input: Embedded context [B, T, C]
        # Output: will be summed up with the MHA output [B, T, C]
        self.ln1 = nn.LayerNorm(embed_dim)

        # 3) Instantiating the FF
        self.ff = FeedForward(embed_dim)

        # 4) Creating the layer normalization module 2
        # Input: Output of MHA [B, T, C]
        # Output: will be summed up with the FF output [B, T, C]
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        # 5) Connecting layer norm 1 and MHA
        x = x + self.mha(self.ln1(x))

        #6) Connecting layer norm 2 and FeedForward
        return x + self.ff(self.ln2(x))
