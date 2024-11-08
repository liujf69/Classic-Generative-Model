# Gan-Demo
# 简单介绍及学习目的
1. 使用 Mnist 数据集进行训练。
```python
# download dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

2. 定义和初始化判别器和生成器。
```python
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

# init discriminator and generator
discriminator = Dis_Model().to(device)
generator = Gen_model().to(device)
```

3. GAN 训练流程：先训练判别器，再训练生成器。
```python
for epoch in range(total_epoch):
    for batch_idx, (batch_imgs, batch_labels) in enumerate(train_loader):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        batch_ = batch_imgs.shape[0]
        batch_real_imgs = batch_imgs.reshape(batch_, -1) # 真实图像
        batch_fake_imgs = generator(batch_) # 使用generator生成虚假图像

        # train discriminator
        pre1 = discriminator(batch_real_imgs)
        loss1 = loss_fun(pre1, torch.ones(batch_, 1, device=device)) # 真实图像，使用标签1

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
```
