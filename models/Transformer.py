import torch.nn as nn
import torch
from torch.nn import functional as F
from .AttentionBlock import AttentionBlock
from tiktoken import get_encoding


class Transformer(nn.Module):
    def __init__(self, embed_dim, n_heads, vocab_size, seq_size, n_layers, device):
        super().__init__()
        self.device = device
        self.seq_size = seq_size
        # self.vocab_size = vocab_size
        self.tokenizer = get_encoding('gpt2')
        self.vocab_size = self.tokenizer.n_vocab

        # 1) Instantiating Token embedding
        self.TokenEmbedding = nn.Embedding(vocab_size, embed_dim) # output: [B,T,C]

        # 2) Instantiating Positional embedding
        self.PositionalEmbedding = nn.Embedding(seq_size, embed_dim) #output: [T,C]

        # 3) Instantiating Attention block
        # Creating a list of Block modules
        block_list = []
        for _ in range(n_layers):
            block = AttentionBlock(embed_dim, n_heads)
            block_list.append(block)
        # Convert the list into a Sequential container
        self.blocks = nn.Sequential(*block_list) # * unpacks the list

        # 4) Last Layer Normalization module
        self.lnn = nn.LayerNorm(embed_dim)

        # 5) Last Linear layer to go from [B,T,C] to [B,T,vocab_size]
        self.linearn = nn.Linear(embed_dim,vocab_size)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text, allowed_special="all")
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens.tolist())
    
    def forward(self,context, targets=None):

        context = context.to(self.device)
        if targets is not None: targets = targets.to(self.device)
        
        self.batch_size = context.shape[0]
        self.seq_size   = context.shape[1]
        #vocab_size from the global

        
        # 1) Token embedding
        tm = self.TokenEmbedding(context) # [B, T, C]

        # 2) Positional embedding
        # create a sequence of int numbers of 0 to vocab_size-1
        tmp = torch.arange(self.seq_size, dtype=torch.int64).to(self.device) 
        pm  = self.PositionalEmbedding(tmp)

        # Adding the two embeddings
        x = tm + pm

        # 3) Calculate the blocks
        x = self.blocks(x)

        # 4 and 5) Pass the data through lnn and last linear layer
        x = self.lnn(x)
        y = self.linearn(x) # output: [B, T, vocab_size]

        # If there are targets, it's training otherwise is inference
        if targets is None:
            loss = None
        else:
            y       = y.view(self.batch_size*self.seq_size, self.vocab_size)
            targets = targets.view(self.batch_size*self.seq_size)
            loss    = F.cross_entropy(y, targets)

        return y, loss
    
    # def generation(self, context, max_tokens):
    #     # context has dimensions of [B, T]
    #     for _ in range(max_tokens):
    #         # make sure the context fits in the sequence length
    #         context_crop = context[:, -self.seq_size:]
    #
    #         # get the predictions
    #         y, _ = self(context_crop)
    #
    #         # focus only on the last token
    #         y = y[:, -1, :] # becomes (B, C)
    #
    #         # apply softmax to get probabilities
    #         probs = F.softmax(y, dim=-1) # (B, C)
    #
    #         # sample from the distribution
    #         next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
    #
    #         # append the sample to the running sequence
    #         context = torch.cat((context, next_token), dim=1) # (B, T+1)
    #     return context

    def generation(self, context, max_tokens: int, temperature: float = 1.0) -> str:
        self.eval()
        with torch.no_grad():
            if isinstance(context, str): context_tokens = self.encode(context).unsqueeze(0)
            else: context_tokens = context
            for _ in range(max_tokens):
                context_crop = context_tokens[:, -self.seq_size:]
                y, _ = self(context_crop)
                y = y[:, -1, :]
                probs = F.softmax(y / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context_tokens = torch.cat((context_tokens, next_token), dim=1)

                # Handle special end-of-sequence token
                end_token = self.tokenizer.encode("<|endoftext|>")[0]
                if next_token.item() == end_token:
                    break
            return self.decode(context_tokens.squeeze(0))
