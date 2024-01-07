import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp

class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.decoder = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                                    ResidualBlock(dim), ResidualBlock(dim),
                                    nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
                                    nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2

    def forward(self, x):
        # encode
        ze = self.encoder(x) # x.shape: B C(3) H(64) W(64) ze.shape: B C(128) H(16) W(16)
        embedding = self.vq_embedding.weight.data # embedding.shape: K(32) C(128)
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1) # 1 K C 1 1
        ze_broadcast = ze.reshape(N, 1, C, H, W) # B 1 C H W
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2) # 基于通道维度C计算距离 
        nearest_neighbor = torch.argmin(distance, 1) # 取距离最近的embedding nearest_neighbor.shape: B H W
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2) # 获取最近的embedding作为zq zq.shape B C(128) H W
        # stop gradient
        decoder_input = ze + (zq - ze).detach() # trick

        # decode
        x_hat = self.decoder(decoder_input) # 重构图像
        return x_hat, ze, zq

    @torch.no_grad() # 不计算梯度，当训练pixelcnn时调用
    def encode(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad() # 不计算梯度，当训练pixelcnn时调用
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)
