# MoE
MoE模型全称是**混合专家模型**（Mixture of Experts, MoE），其主要将多个专家神经网络模型组合成一个更大的模型。  
MoE模型的核心组成有两部分：  
&emsp; 第一部分是**多个专家网络模型**，每个专家网络模型往往是独立的，且分别用于不同的问题；  
&emsp; 第二部分是**门控网络**，用于确定使用哪些专家网络模型，一般通过计算每个专家网络的分数（权重）来实现。

# Demo
```python
# 定义MoE模型
class MoE(nn.Module):
    def __init__(self, num_experts, intput_size, output_size):
        super(MoE, self).__init__()
        # 专家模型数
        self.num_experts = num_experts
        # 初始化多个专家模型
        self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(self.num_experts)])
        self.gating_network = nn.Linear(input_size, num_experts)
        
    def forward(self, x):
        # 门控网络决定权重
        gating_scores = F.softmax(self.gating_network(x), dim = 1) # [Batchsize, num_experts]
        # 获取每个专家网络的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim = 1) # [Batchsize, num_experts, output_size]
        # 专家网络的结果进行加权融合，获取最终输出
        moe_output = torch.bmm(gating_scores.unsqueeze(1), expert_outputs).squeeze(1) # [Batchsize, output_size]
        return moe_output
```
