import torch.nn as nn
import torch
from torch.nn import functional as F
from SingleHeadAttention import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,n_heads):
        super().__init__()
        
        # 3a) Creating a list of SingleHeadAttention
        # Not just a simple list, but a ModuleList. 
        # It allows the model to track the parameters
        head_size = embed_dim//n_heads
        self.heads = nn.ModuleList([SingleHeadAttention(embed_dim,head_size) 
                                for _ in range(n_heads)])
        
        # 4) Adding an additional linear layer (called projection)
        # Will take the concatanated attentions [B,T,C] 
        # Its output should have the same dimension as the embedded context [B, T, C]
        self.proj  = nn.Linear(embed_dim,embed_dim)

        # 5) Adding the residual with dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,context):

        # 3b) concatanating the list of SingleHeads modules
        heads_list=[]
        for head in self.heads:
            heads_list.append(head(context))
        heads = torch.cat(heads_list,dim=-1)
        #print(f'heads.shape={heads.shape}\n ---')

        # 4) adding the linear layer, called projection
        scores=self.proj(heads) # now score is [B, T, C]

        # 5) adding the dropout layer
        scores = self.dropout(scores)

        return scores #[B, T, C]