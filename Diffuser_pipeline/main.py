import time
import numpy as np
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from matplotlib import pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline

# 数据增广
def transform(examples):
    preprocess = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

def process_dataset(batch_size):
    # 加载数据集
    # dataset = load_dataset("huggan/smithsonian_butterflies_subset", split = "train")
    dataset = load_dataset("/data-home/liujinfu/Diffuser/Data/smithsonian_butterflies_subset", split = "train")
    # 调用自定义的transform函数
    dataset.set_transform(transform)
    # 设置dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = True
    )
    return train_dataloader

def train_loop(train_dataloader, noise_scheduler, model, num_epoches, device):
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr = 4e-4)
    losses = []
    start_time = time.time() 
    for epoch in range(num_epoches):
        for _, batch in enumerate(train_dataloader): # 遍历
            clean_images = batch["images"].to(device) # B C H W
            # sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device) # B C H W
            bs = clean_images.shape[0] # 64

            # sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs, ), device = clean_images.device
            ).long() # B

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps) # 加噪

            # Get model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise) # 计算预测噪音和真实噪音之间的损失
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

    end_time = time.time()
    elapsed_time = end_time - start_time # 记录训练时间
    
    print("time cost: ", elapsed_time)
    return losses

def vis(losses):
    # 可视化 loss
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    return fig

def generate(model, noise_scheduler):
    # 1. create a pipeline
    image_pipe = DDPMPipeline(unet = model, scheduler = noise_scheduler)

    pipeline_output = image_pipe()
    return pipeline_output.images[0]

# 可视化生成图像
def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im

def main():
    # 获取训练集
    image_size = 32 
    batch_size = 64
    train_dataloader = process_dataset(batch_size = batch_size)

    # 设置Scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps = 1000, beta_schedule = "squaredcos_cap_v2") 

    # 创建Unet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DModel(
        sample_size = image_size, # target image resolution
        in_channels = 3,
        out_channels = 3,
        layers_per_block = 2, # how many resnet layers to use per Unet block
        block_out_channels = (64, 128, 128, 256),
        down_block_types = (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D", # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D", # a regular ResNet upsampling block
        ),  
    ).to(device)
    
    # 开始训练
    losses = train_loop(train_dataloader = train_dataloader, 
               noise_scheduler = noise_scheduler,
               model = model,
               num_epoches = 30, 
               device = device)
    
    fig = vis(losses)
    fig.savefig("./loss.png")
    
    # 生成一张图片
    gen_img = generate(model, noise_scheduler)
    gen_img.save("./generate1.png")
    
    # 随机初始化噪音生成图片
    sample = torch.randn(8, 3, 32, 32).to(device)
    for i, t in enumerate(noise_scheduler.timesteps): # 反向去噪
        # Get model pred
        with torch.no_grad():
            residual = model(sample, t).sample
        # Update sample with step
        sample = noise_scheduler.step(residual, t, sample).prev_sample
    
    # 可视化生成的图片
    grid_im = show_images(sample)
    grid_im.save("./genearate2.png")
    print("All Done!")

if __name__ == "__main__":
    main()
