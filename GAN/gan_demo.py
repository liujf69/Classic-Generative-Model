import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# discriminator
class Dis_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784,2000)
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(2000,1)

    def forward(self,x):
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        return x
    
# generator
class Gen_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512,1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024,1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024,784)

    def forward(self, batch_size): 
        x = torch.randn(size=(batch_size, 512),device=device)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

# save images
def save_img(imgs, e):
    imgs = imgs.reshape(imgs.shape[0],1,28,28)
    imgs = imgs.clamp(0,1)
    imgs = torchvision.utils.make_grid(imgs, nrow=8).detach().cpu().numpy()
    plt.imshow(imgs.transpose(1,2,0))
    plt.savefig(f"./result/{e}.jpg")

if __name__ == "__main__":
    # download dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # init batch_size and epochs
    batch_size = 200
    total_epoch = 100

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init discriminator and generator
    discriminator = Dis_Model().to(device)
    generator = Gen_model().to(device)

    # init loss function
    loss_fun = nn.BCEWithLogitsLoss()

    # init optimizer
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0001)

    for epoch in range(total_epoch):
        for batch_idx, (batch_imgs, batch_labels) in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            batch_ = batch_imgs.shape[0]
            batch_real_imgs = batch_imgs.reshape(batch_, -1) # 真实图像
            batch_fake_imgs = generator(batch_) # 生成虚假图像

            # train discriminator
            pre1 = discriminator(batch_real_imgs)
            loss1 = loss_fun(pre1,torch.ones(batch_, 1, device=device)) # 真实图像，使用标签1

            pre2 = discriminator(batch_fake_imgs)
            loss2 = loss_fun(pre2, torch.zeros(batch_, 1, device=device)) # 虚假图像，使用标签0

            loss_d = loss1 + loss2 # 通过loss1和loss2训练辨别器
            loss_d.backward()
            opt_d.step()
            opt_d.zero_grad()
            opt_g.zero_grad()

            # train generator
            batch_fake_imgs_2 = generator(batch_)
            pre3 = discriminator(batch_fake_imgs_2)
            loss_g = loss_fun(pre3, torch.ones(batch_, 1, device=device)) # 为了训练生成器，使用虚假图像，并使用标签1来欺骗辨别器

            loss_g.backward()
            opt_g.step()
            opt_d.zero_grad()
            opt_g.zero_grad()

        print('epoch: ', epoch, 'loss_g: ', loss_g.item(), "loss_d: ", loss_d.item())

        imgs = generator(32) # 每个epoch之后，利用生成器生成32个样本
        save_img(imgs, epoch) # 保存生成的图片
