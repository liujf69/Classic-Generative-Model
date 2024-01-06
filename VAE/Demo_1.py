import os
import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from VAE_1 import VAE

class process():
    def __init__(self, device, image_size, h_dim, z_dim, num_epochs, 
                    batch_size, learning_rate, data_loader, sample_dir):
        super(process, self).__init__()
        self.device = device
        self.image_size = image_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = VAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        self.data_loader = data_loader
        self.sample_dir = sample_dir
    
    def train(self):
        for epoch in range(self.num_epochs):
            # training 
            for i, (x, _) in enumerate(self.data_loader):
                # Forward pass
                x = x.to(self.device).view(-1, self.image_size)
                x_reconst, mu, log_var = self.model(x)
                # Compute reconstruction loss and kl divergence
                reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                # Backprop and optimize
                loss = reconst_loss + kl_div
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 200 == 0:
                    print("Epoch[{}/{}], Step [{}/{}], total Loss: {:.4f}=Reconst Loss: {:.4f}+KL Div: {:.4f}" 
                        .format(epoch + 1, self.num_epochs, i + 1, len(self.data_loader), 
                                loss.item(), reconst_loss.item(), kl_div.item()))
            # sampling/generating 
            with torch.no_grad():
                # 从标准正态分布中进行采样
                z = torch.randn(self.batch_size, z_dim).to(self.device)
                out = self.model.decode(z).view(-1, 1, 28, 28)
                # 16*8
                save_image(out, os.path.join(self.sample_dir, 'sampled-{}.png'.format(epoch + 1)))
                # Save the reconstructed images
                out, _, _ = self.model(x)
                x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim = 3)
                # 16*16
                save_image(x_concat, os.path.join(self.sample_dir, 'reconst-{}.png'.format(epoch+1)))

if __name__ == "__main__":
    # 利用tensorboard可视化模型
    # writer = SummaryWriter('runs/model_visualization')
    # vae = VAE()
    # inputs = torch.randn(1, 28*28)
    # writer.add_graph(vae, inputs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_size = 784 # 28*28
    h_dim = 400
    z_dim = 20
    num_epochs = 20
    batch_size = 128
    learning_rate = 3e-4

    sample_dir = './samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # FashionMNIST dataset
    dataset = datasets.FashionMNIST(root='./data', 
                         train=True, 
                         transform=transforms.ToTensor(), 
                         download=True)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)
    
    mytest = process(device, image_size, h_dim, z_dim, num_epochs, 
                     batch_size, learning_rate, data_loader, sample_dir)
    
    mytest.train()

    print("All done!")