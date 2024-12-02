import torch.nn as nn
import torch
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()

        ### 1) q, k, v linear transformation matrices
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,context):
        q = self.query(context) # dimension: (B, T, C)
        k = self.key(context)   # dimension: (B, T, C)
        v = self.value(context) # dimension: (B, T, C)

        ### 2) Q @ K^T
        # We need to keep the first dimension to be the batch size: "B"
        # PyTorch will be able to perform matmul in parallel (at different batch indices)
        # We need to transpose the other two indices (-2 and 01)
        
        print('q.shape,k.shape=',q.shape,k.shape)
        k_transposed = k.transpose(-2, -1) #swaps the dim "-2" with dim "-1"
        print('k_transposed.shape=',k_transposed.shape)
        
        ### 2-3) scaling the attention scores
        embed_dim = q.shape[-1]
        scores = q @ k_transposed * (embed_dim ** -0.5)  # Scale by sqrt(d_k) # [B, T, T]

        ### 4) Masking
        # We want to mask the next tokens from affecting the current token
        # at i, we have to mask j>i --> similar to a lower triangular matrix
        # We set them to -inf so after softmax, they become 0
        
        print('---')
        print('scores at B=0, before masking: \n', scores[0,:,:],'\n ---')
        seq_size = q.shape[-2]
        self.tril=torch.tril(torch.ones(seq_size, seq_size))
        
        scores = scores.masked_fill(
            self.tril[:seq_size, :seq_size] == 0, float('-inf')
            ) # (B, T, T)
        print('scores at B=0, after masking: \n', scores[0,:,:],'\n ---')

        ### 5) applying the softmax
        # scores has the dimension [B, T, T]
        # we want to normalize the score for each query position independently
        # so, softmax is applied along the last dimension

        scores = F.softmax(scores, dim=-1) #[B, T, T]
        print('scores at B=0, after softmax: \n', scores[0,:,:],'\n ---')

        ### 6) multiplying by v matrix: [B, T, C]
        scores = scores @ v

        return scores # B, T, C]

""" 
SA=SelfAttention(embed_dim)
scores=SA(org_embd_context)         
"""