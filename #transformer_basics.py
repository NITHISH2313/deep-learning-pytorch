#transformer_basics

import torch
import torch.nn as nn
import math

#=====attention mechanism======

class SelfAttention(nn.Module):
    def __init__(self , embed_size , heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size

        # query , key , value projections
        self.query = nn.Linear(embed_size,embed_size)
        self.key = nn.Linear(embed_size,embed_size)
        self.value= nn.Linear(embed_size,embed_size)
        self.fc = nn.Linear(embed_size,embed_size)
    
    def forward(self , x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        #create Q , K , V
        Q = self.query(x)
        K =self.key(x)
        V = self.value(x)

        # calculate attention scores
        scores = torch.matmul(Q,K.transpose(-2 , -1))
        scores = scores / math.sqrt(self.head_dim)
        attention = torch.softmax(scores ,dim = -1)

        #apply attention to values

        out  = torch.matmul(attention ,V)
        out = self.fc(out)
        return out , attention
    
#=====simple transformer block======
class TransformerBlock(nn.Module):
    def __init__(self , embed_size , heads):
        super().__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(nn.Linear(embed_size,embed_size*4),nn.ReLU(),nn.Linear(embed_size*4 ,embed_size))
    def forward(self , x):
        #attention + residual connection
        att_out,attention = self.attention(x)
        x = self.norm1(x+att_out)
        #fedd forward + residual connection
        ff_out = self.ff(x)
        x = self.norm2(x+ff_out)
        return x,attention
    

#====test=====
embed_size = 64
heads =4
seq_len = 5
batch = 1

# create random input
x = torch.randn(batch,seq_len,embed_size)
print(f"input shape:{x.shape}")

#create transformer block
transformer = TransformerBlock(embed_size,heads)
output ,attention = transformer(x)

print(f"output shape :{output.shape}")
print(f"attention shape :{attention.shape}")
print(f"\n attention weights(first sequence):")
print(attention[0].detach().numpy().round(3))
print(f"\n each row shows how much each word")
print(f"attends to every other word")




        