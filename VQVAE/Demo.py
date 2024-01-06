import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange
import umap
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from VQ_VAE import Model

def show(img, save_img = None):
    npimg = img.numpy()
    fig = plt.figure(figsize=(16,8))
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('./output/' + save_img)

class Process():
    def __init__(self, batch_size = 256, num_training_updates = 1500, num_hiddens = 128, num_residual_layers = 2, num_residual_hiddens = 32,
                    num_embeddings = 512, embedding_dim = 64, commitment_cost = 0.25, decay = 0.99, device = "cuda", learning_rate = 1e-3):
        super(Process, self).__init__()
        self.batch_size = batch_size
        self.num_training_updates = num_training_updates
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.device = device
        self.learning_rate = learning_rate
        self.load_dataset()
        self.load_model()
        self.data_variance = np.var(self.training_data.data / 255.0)

    def load_dataset(self):
        self.training_data = datasets.CIFAR10(root = "./data", train = True, download = True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]))

        self.validation_data = datasets.CIFAR10(root = "./data", train = False, download = True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))]))
        
        self.training_loader = DataLoader(self.training_data, 
                             batch_size = self.batch_size, 
                             shuffle = True,
                             pin_memory = True)
    
        self.validation_loader = DataLoader(self.validation_data,
                                batch_size = 32,
                                shuffle = True,
                                pin_memory = True)
        
    def load_model(self):
        self.model = Model(self.num_hiddens, self.num_residual_layers, self.num_residual_hiddens,
                    self.num_embeddings, self.embedding_dim, 
                    self.commitment_cost, self.decay).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, amsgrad=False)

    def train(self):
        self.model.train()
        train_res_recon_error = []
        train_res_perplexity = []

        for i in xrange(self.num_training_updates):
            (data, _) = next(iter(self.training_loader))
            data = data.to(self.device)
            self.optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = F.mse_loss(data_recon, data) / self.data_variance # 计算源图像和重建图像的重建损失
            loss = recon_error + vq_loss
            loss.backward()

            self.optimizer.step()
            
            train_res_recon_error.append(recon_error.item()) # 记录重建损失
            train_res_perplexity.append(perplexity.item())

            if (i+1) % 100 == 0:
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
                print()

        train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
        train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')
        ax = f.add_subplot(1,2,2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('iteration')
        if not os.path.exists('./output/'):
            os.makedirs('./output/')
        plt.savefig('./output/metric.png')
    
    def eval(self):
        self.model.eval()
        (valid_originals, _) = next(iter(self.validation_loader)) # 取32个样本进行测试
        valid_originals = valid_originals.to(self.device)

        vq_output_eval = self.model._pre_vq_conv(self.model._encoder(valid_originals))
        _, valid_quantize, _, _ = self.model._vq_vae(vq_output_eval)
        valid_reconstructions = self.model._decoder(valid_quantize)

        show(make_grid(valid_reconstructions.cpu().data)+0.5, 'reconstructions.png') # 展示重建图像
        show(make_grid(valid_originals.cpu()+0.5), 'originals.png') # 展示源图像

        # 绘制embedding space
        proj = umap.UMAP(n_neighbors = 3, min_dist = 0.1,
                    metric='cosine').fit_transform(self.model._vq_vae._embedding.weight.data.cpu())
        fig = plt.figure(figsize=(16, 8))
        plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        plt.savefig("./output/Embedding.png")

    def run(self):
        self.train()
        self.eval()

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    num_training_updates = 1500
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3

    mytest = Process(batch_size, num_training_updates, num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, commitment_cost, decay, device, learning_rate)
    
    mytest.run()

if __name__ == "__main__":
    main()
    print("All done!")
