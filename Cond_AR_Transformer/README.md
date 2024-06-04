# Conditional Autoregressive Transformer
1. 自回归Transformer的核心在于每个Token**只能看到自身和前面的Token**，因此需要生成合适的Attention Mask。
```python
# create two attention mask (actually they have the same function)
attention_mask1 = torch.triu((torch.ones((seq_length, seq_length)) == 1), diagonal=1) # bool type
attention_mask2 = attention_mask1.float() # True->1 False->0
attention_mask2 = attention_mask2.masked_fill(attention_mask2 == 1, float("-inf"))  # Convert ones to -inf
```
2. 通过Cross Attention机制实现条件驱动，以文生图为例，其中噪声图作为Query，文本作为Key和Value；
```python
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
```

# Run
```python
python Demo.py
```
