import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # 论文中损失函数的第三项
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # 论文中损失函数的第二项
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach() # 梯度复制
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float, epsilon: float = 1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs: torch.Tensor):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape # B(256) H(8) W(8) C(64)
        
        # Flatten input BHWC -> BHW, C
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances 计算与embedding space中所有embedding的距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # 取最相似的embedding
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # 映射为 one-hot vector
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # 根据index使用embedding space对应的embedding
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw) 
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1)) # 论文中公式(8)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # 计算encoder输出（即inputs）和decoder输入（即quantized）之间的损失
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach() # trick, 将decoder的输入对应的梯度复制，作为encoder的输出对应的梯度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class Residual(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_hiddens: int):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels = in_channels,
                      out_channels = num_residual_hiddens,
                      kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(in_channels = num_residual_hiddens,
                      out_channels = num_hiddens,
                      kernel_size = 1, stride = 1, bias = False)
        )
    
    def forward(self, x: torch.Tensor):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x: torch.Tensor):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Conv + Residual
class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels = in_channels,
                                 out_channels = num_hiddens//2,
                                 kernel_size = 4,
                                 stride = 2, 
                                 padding = 1)
        self._conv_2 = nn.Conv2d(in_channels = num_hiddens//2,
                                 out_channels = num_hiddens,
                                 kernel_size = 4,
                                 stride = 2, 
                                 padding = 1)
        self._conv_3 = nn.Conv2d(in_channels = num_hiddens,
                                 out_channels = num_hiddens,
                                 kernel_size = 3,
                                 stride = 1, 
                                 padding = 1)
        self._residual_stack = ResidualStack(in_channels = num_hiddens,
                                             num_hiddens = num_hiddens,
                                             num_residual_layers = num_residual_layers,
                                             num_residual_hiddens = num_residual_hiddens)

    def forward(self, inputs: torch.Tensor):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)

# Conv + Residual + ConvTranspose2d
class Decoder(nn.Module):
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels = in_channels,
                                 out_channels = num_hiddens,
                                 kernel_size = 3, 
                                 stride = 1, 
                                 padding = 1)
        self._residual_stack = ResidualStack(in_channels = num_hiddens,
                                             num_hiddens = num_hiddens,
                                             num_residual_layers = num_residual_layers,
                                             num_residual_hiddens = num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels = num_hiddens, 
                                                out_channels = num_hiddens//2,
                                                kernel_size = 4, 
                                                stride = 2, 
                                                padding = 1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels = num_hiddens//2, 
                                                out_channels = 3,
                                                kernel_size = 4, 
                                                stride = 2, 
                                                padding = 1)

    def forward(self, inputs: torch.Tensor):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)

class Model(nn.Module):
    def __init__(self, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int, 
                 num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float = 0.0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(in_channels = 3, 
                                num_hiddens = num_hiddens,
                                num_residual_layers = num_residual_layers, 
                                num_residual_hiddens = num_residual_hiddens)
        
        self._pre_vq_conv = nn.Conv2d(in_channels = num_hiddens, 
                                      out_channels = embedding_dim,
                                      kernel_size = 1, 
                                      stride = 1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings = num_embeddings, 
                                              embedding_dim = embedding_dim, 
                                              commitment_cost = commitment_cost, 
                                              decay = decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings = num_embeddings, 
                                           embedding_dim = embedding_dim,
                                           commitment_cost = commitment_cost)
        
        self._decoder = Decoder(in_channels = embedding_dim,
                                num_hiddens = num_hiddens, 
                                num_residual_layers = num_residual_layers, 
                                num_residual_hiddens = num_residual_hiddens)

    def forward(self, x: torch.Tensor): # x.shape: B(256) C(3) H(32) W(32)
        z = self._encoder(x) # B(256) C(128) H(8) W(8)
        z = self._pre_vq_conv(z) # B(256) C(64) H(8) W(8)
        loss, quantized, perplexity, _ = self._vq_vae(z) # VQ # quantized.shape: B(256) C(64) H(8) W(8)
        x_recon = self._decoder(quantized) # B(256) C(3) H(32) W(32)

        return loss, x_recon, perplexity
