import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.proj_K = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.proj_Q = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.proj_V = nn.Linear(embedding_dim,attention_dim,bias=False)
        

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.proj_K(embedded)
        q = self.proj_Q(embedded)
        v = self.proj_V(embedded)
        scores = q @ torch.transpose(k,1,2)
        
        context_dim,attention_dim = k.shape[1],k.shape[2]
        scores = scores / (attention_dim ** 0.5)
        lower_tril = torch.tril(torch.ones(context_dim,context_dim))# 下三角为 1，上三角为 0
        mask = lower_tril==0    # 取反：上三角（未来位置）为 True
        scores = scores.masked_fill(mask,float('-inf'))
        scores = nn.functional.softmax(scores, dim=-1)
        return torch.round(scores @ v, decimals=4)
