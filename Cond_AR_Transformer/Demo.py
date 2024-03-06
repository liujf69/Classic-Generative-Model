import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, input_x: torch.Tensor, condition: torch.Tensor, attn_mask: torch.Tensor = None):
        '''
        query: input_x
        key: condition
        val: condition
        '''
        input_x = self.cross_attn(input_x, condition, condition, attn_mask=attn_mask)[0]
        return input_x

class Cond_Autoregressive_layer(nn.Module):
    def __init__(self, input_dim: int, condtion_dim: int, embed_dim: int, num_heads: int):
        super(Cond_Autoregressive_layer, self).__init__()
        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.linear2 = nn.Linear(condtion_dim, embed_dim)
        self.cond_multihead_attn = CrossAttention(embed_dim = embed_dim, num_heads = num_heads)
    
    def forward(self, input_x: torch.Tensor, conditon: torch.Tensor, attention_mask1: torch.Tensor, attention_mask2: torch.Tensor):
        # q, k, v, attention mask, here we set key and value are both condtion 
        y1 = self.cond_multihead_attn(self.linear1(input_x), self.linear2(conditon), attn_mask = attention_mask1)
        y2 = self.cond_multihead_attn(self.linear1(input_x), self.linear2(conditon), attn_mask = attention_mask2)
        return y1, y2
 
if __name__ == "__main__":
    # set sequence len, embedding dim, multi attention head
    seq_length = 10
    input_dim = 32
    condtion_dim = 128
    embed_dim = 64
    num_heads = 8
 
    # init input sequence and condtion
    input_x = torch.randn(seq_length, 1, input_dim)
    condtion = torch.randn(seq_length, 1, condtion_dim)

    # create two attention mask (actually they have the same function)
    attention_mask1 = torch.triu((torch.ones((seq_length, seq_length)) == 1), diagonal=1) # bool type
    attention_mask2 = attention_mask1.float() # True->1 False->0
    attention_mask2 = attention_mask2.masked_fill(attention_mask2 == 1, float("-inf"))  # Convert ones to -inf
 
    # init model
    AG_layer = Cond_Autoregressive_layer(input_dim, condtion_dim, embed_dim, num_heads)
    # forward
    y1, y2 = AG_layer(input_x, condtion, attention_mask1, attention_mask2)
    
    # here we demonstrate the attention_mask1 and attention_mask2 have the same function
    assert(y1[0].equal(y2[0]))