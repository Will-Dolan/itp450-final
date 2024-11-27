import torch.nn as nn
import torch
from torch.nn import functional as F

# 1) Specify the number of heads
# TODO: move this to model building
n_heads = 5
head_size = embed_dim // 5

dropout = 0. # The dropout fraction!

# 2) Single head attention
class SingleHeadAttention(nn.Module):
    def __init__(self,embed_dim,head_size):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key   = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

    def forward(self,context):
        q = self.query(context) # dimension: [B, T, C']
        k = self.key(context)   # dimension: [B, T, C']
        v = self.value(context) # dimension: [B, T, C']

        seq_size  = q.shape[-2]
        embed_dim = q.shape[-1]

        k_transposed = k.transpose(-2, -1) #swaps the dim "-2" with dim "-1"
        scores = q @ k_transposed * (embed_dim ** -0.5)  # Scale by sqrt(d_k) # [B, T, T]
        
        self.tril=torch.tril(torch.ones(seq_size, seq_size)).to(scores.device)
        scores = scores.masked_fill(
            self.tril[:seq_size, :seq_size] == 0, float('-inf')
            ) # (B, T, T)
        
        scores = F.softmax(scores, dim=-1) #[B, T, T]
        scores = scores @ v #[B, T, T] x [B, T, C'] = [B, T, C']

        return scores #[B, T, C']