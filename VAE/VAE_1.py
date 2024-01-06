import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, input_dim = 784, h_dim = 400, z_dim = 20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim) # 28*28 → 784
        self.fc21 = nn.Linear(h_dim, z_dim) # 均值
        self.fc22 = nn.Linear(h_dim, z_dim) # 标准差
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim) # 784 → 28*28
        self.input_dim = input_dim

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1) # 均值、标准差

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # z = mu + eps*std
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # sigmoid: 0-1 之间，后边会用到 BCE loss 计算重构 loss（reconstruction loss）
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar