import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
           
import random
import imageio
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import einops
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Lambda

from UNet import MyUNet
from DDPM import MyDDPM

def show_images(images, title = ""):
    save_img_path = './save_img/'
    if type(images) is torch.Tensor: # Tensor To Numpy
        images = images.detach().cpu().numpy()

    # 设置画布
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)
            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize = 30)

    plt.savefig(save_img_path + title + '.png')
    # plt.show()

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Origin images in the first batch")
        break

def show_forward(ddpm, loader, device):
    # 展示前向加噪过程
    for batch in loader:
        imgs = batch[0]
        show_images(imgs, "Original images before add noise") # 展示原始图片
        for percent in [0.25, 0.5, 0.75, 1]: # 展示4阶段的加噪结果
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break # 只展示第一个batch

def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=28, w=28):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # 随机初始化噪声
        x = torch.randn(n_samples, c, h, w).to(device)
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()
                x = x + sigma_t * z
            
            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # 记录去噪过程中的每一帧
                frames.append(frame)

    # 将过程保存为 gif 图像
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            # Convert grayscale frame to RGB
            rgb_frame = np.repeat(frame, 3, axis=-1)
            writer.append_data(rgb_frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(rgb_frame)
    return x

def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0) # batch_size

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # 展示每一个epoch的生成能力
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

def init_dataset(fashion):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )

    ds_fn = FashionMNIST if fashion else MNIST
    dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    return loader

if __name__ == "__main__":
    # 设置种子
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 设置显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

    # 设置训练参数
    train_flag = False # 是否重新训练模型（第一次使用需要设置为True）
    fashion = True # True 时使用 fashion 数据集，False 时使用 mnist 数据集
    batch_size = 128
    n_epochs = 2
    lr = 0.001

    # DDPM
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # 设置DDPM的参数
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

    # 初始化使用的数据集
    loader = init_dataset(fashion)
    # 展示第一个batch用到的图像
    show_first_batch(loader)

    # 可视化前向加噪的过程
    show_forward(ddpm, loader, device)
    generated = generate_new_images(ddpm, gif_name = "before_training.gif") # 展示未经过训练的生成能力
    show_images(generated, "Images generated before training")

    # 开始训练模型
    store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"
    if train_flag:
        training_loop(ddpm, loader, n_epochs, optim = Adam(ddpm.parameters(), lr), device = device, store_path = store_path)

    # 加载最佳模型
    best_model = MyDDPM(MyUNet(), n_steps = n_steps, device = device)
    best_model.load_state_dict(torch.load(store_path, map_location = device))
    best_model.eval()
    print("Model loaded")
    # 展示最佳模型的生成能力
    print("Generating new images")
    generated = generate_new_images(
        best_model,
        n_samples = 100,
        device = device,
        gif_name = "fashion.gif" if fashion else "mnist.gif"
    )

    # 展示最终的结果
    show_images(generated, "Final result")
    print("All done!")