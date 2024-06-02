import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义专家模型
class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.fc(x)

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

if __name__ == "__main__":
    # 定义测试参数
    input_size = 8
    output_size = 64
    num_experts = 4
    
    # 初始化MoE模型
    moe_model = MoE(num_experts, input_size, output_size)
    
    # 初始化输入测试
    batchsize = 2
    input = torch.randn(batchsize, input_size)
    
    # 推理
    output = moe_model(input)
    print("output.shape: ", output.shape) # [batchsize, output_size]
        