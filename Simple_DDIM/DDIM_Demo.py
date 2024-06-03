import torch
import torch.nn as nn
import numpy as np
 
# 简化的扩散模型
class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model, self).__init__()
    def forward(self, x, t):
        # 实际的模型会预测x在时间步t的噪声，例如UNet模型
        return torch.randn_like(x)
 
# ddim的实现
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0) # beta -> [1, beta]
    # 先通过cumprod计算累乘结果，即: alpha_(t)_hat = alpha_(t) * alpha_(t-1) * ... * alpha_1 * alpha_0
    # 再选取alpha_(t)_hat, 这里用索引t+1来选取
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# ddim的实现, 参考: https://github.com/ermongroup/ddim/blob/main/functions/denoising.py
def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0) # batchsize
        seq_next = [-1] + list(seq[:-1]) # t-skip: [-1, 0, 10, 20, ..., 980], len: 100
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)): # i = t, j = t-skip
            t = (torch.ones(n) * i).to(x.device) # t
            next_t = (torch.ones(n) * j).to(x.device) # t-1
            at = compute_alpha(b, t.long()) # alpha_(t)_hat
            at_next = compute_alpha(b, next_t.long()) # alpha_(t-1)_hat
            xt = xs[-1].to('cuda') # 获取当前时间步的样本，即x_t
            et = model(xt, t) # 预测噪声
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() # 论文公式(12)中的 predicted x0
            x0_preds.append(x0_t.to('cpu')) # 记录当前时间步的 predicted x0
            c1 = (kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()) # 计算公式(12)中的标准差(\sigma)_(t)
            c2 = ((1 - at_next) - c1 ** 2).sqrt() # 论文公式(12)中 direction pointing to xt 的系数
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et # 根据公式(12)计算x_(t-1)
            xs.append(xt_next.to('cpu')) # 记录每一个时间步的x_(t-1)

    return xs, x0_preds # 保存了每一个时间步的结果

if __name__ == "__main__":
    # 初始化给定时间步，预测噪声的模型
    model = diffusion_model().cuda()

    # 初始化一个噪声样本x_noise
    x_noise = torch.randn(1, 3, 64, 64).cuda() # [batchsize, channel, height, width]

    # 使用 uniform 生成时间步
    num_timesteps = 1000 # 初始化时间步
    skip = 10 # 初始化间隔10步 (ddpm则为1)
    seq = range(0, num_timesteps, skip) # [0, 10, 20, ..., num_timesteps - 10]

    # 初始化噪声参数betas, ddim实际初始化方法参考: https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py#L80
    betas = np.linspace(0.0001, 0.02, num = 1000)
    betas = torch.from_numpy(betas).cuda()

    # 运行ddim
    xs, x0_preds = generalized_steps(x_noise, seq, model, betas)
    print("len(xs): ", len(xs)) # 101
    print("len(x0_preds): ", len(x0_preds)) # 100

    # 实际上我们只需要最后一个时间步的生成结果
    print("output.shape: ", xs[-1].shape) # [batchsize, channel, height, width]