from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from Unet import ContextUnet

# 根据min_beta, max_beta和扩散步长T来生成一系列参数
def ddpm_schedules(beta1: float = 0.0001, beta2: float = 0.02, T: int = 1000):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

# DDPM类
class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob = 0.1):
        super(DDPM, self).__init__()
        # 初始化UNet模型
        self.nn_model = nn_model.to(device)

        # 根据min_beta, max_beta和扩散步长T注册参数
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        Args:
            x: 输入的图片, shape: [batchsize, 1, 28, 28]
            c: 输入的类别, shape: [batchsize]
        """
        # 随机选取步长t和噪声 
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T) shape: [batchsize]
        noise = torch.randn_like(x)  # eps ~ N(0, 1) shape: [batchsize, 1, 28, 28]

        # 基于DDPM公式进行加噪
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # 通过伯努利分布来生成0和1值，从而生成dropout矩阵
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)
        
        # 基于真实噪声和基于UNet预测噪声之间的损失
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    # Classifier-Free Diffusion Guidance
    # 可以参考https://sunlin-ai.github.io/2022/06/01/Classifier-Free-Diffusion.html
    # 具体公式有区别，但思路是一样的，通过guide_w来动态考虑有条件和无条件生成的结果
    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise # 随机采样噪声 [num_sample, 1, 28, 28]
        c_i = torch.arange(0, 10).to(device) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] mnist对应的10个类别标签
        c_i = c_i.repeat(int(n_sample/c_i.shape[0])) # 重复n_sample/c_i.shape[0]次, [num_sample]

        # 生成时不dropout类别标签
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2) # 重复两次
        context_mask[n_sample:] = 1. # 前一半为0, 后一半为1（分别对应于有类别指导和没有类别指导）

        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in range(self.n_T, 0, -1): # 去噪
            print(f'sampling timestep {i}', end = '\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1) # [num_sample, 1, 1, 1]

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1) # noise # [2*num_sample, 1, 28, 28]
            t_is = t_is.repeat(2, 1, 1, 1) # [2*num_sample, 1, 1, 1]

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0 # [num_sample, 1, 28, 28]

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask) # [2*num_sample, 1, 28, 28]
            eps1 = eps[:n_sample] # 有类别指导的预测噪音 # [num_sample, 1, 28, 28]
            eps2 = eps[n_sample:] # 没有类别指导的预测噪音 # [num_sample, 1, 28, 28]
            eps = (1 + guide_w) * eps1 - guide_w * eps2 # 根据指导概率 # [0.0, 0.5, 2.0]
            x_i = x_i[:n_sample]
            # 基于DDPM公式进行去噪
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8: # 保存相应去噪步骤的图片
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

def train_mnist():
    # 设置参数
    n_epoch = 20
    batch_size = 256
    n_T = 400 
    device = "cuda:4"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    # 初始化模型
    ddpm = DDPM(nn_model = ContextUnet(in_channels = 1, n_feat = n_feat, n_classes = n_classes), 
                betas = (1e-4, 0.02), 
                n_T = n_T, 
                device = device, 
                drop_prob = 0.1)
    ddpm.to(device)

    # 初始化数据集
    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST("./data", train = True, download = True, transform = tf) # 下载到当前路径的data文件夹
    # 初始化dataloader
    dataloader = DataLoader(dataset, 
                            batch_size = batch_size, 
                            shuffle = True, 
                            num_workers = 5)
    
    # 初始化优化器
    optim = torch.optim.Adam(ddpm.parameters(), lr = lrate)

    # 训练
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device) # 图片
            c = c.to(device) # 条件
            loss = ddpm(x, c) # 损失
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # 生成
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes # 40
            for _, w in enumerate(ws_test): # [0.0, 0.5, 2.0]
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w = w) # x_gen: 生成的图片 x_gen_store: 去噪过程中保存的图片

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5 == 0 or ep == int(n_epoch - 1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval = 200, blit = False, repeat = True, frames = x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi = 100, writer = PillowWriter(fps = 5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
                    
        # 保存模型
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()