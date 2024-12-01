import os
import time
import cv2
import einops
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.configs import get_cfg
from utils.dataset import get_dataloader
from model import VQVAE
from pixelcnn_model import PixelCNNWithEmbedding

# train VQ-VAE
def train_vqvae(model: VQVAE, img_shape = None, device = 'cuda', ckpt_path = './model.pth', batch_size = 64,
                dataset_type = None, lr = 1e-3, n_epochs = 100, l_w_embedding = 1, l_w_commitment = 0.25):
    
    print('Start training vqvae, batch size:', batch_size) # 64
    dataloader = get_dataloader(dataset_type, batch_size, img_shape = img_shape) # dataloader
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    for e in range(n_epochs): # epoch
        total_loss = 0
        for x in dataloader: # load data
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, ze, zq = model(x)
            
            l_reconstruct = mse_loss(x, x_hat) # first item in loss function
            l_embedding = mse_loss(ze.detach(), zq) # second item in loss function
            l_commitment = mse_loss(ze, zq.detach()) # third item in loss function
            loss = l_reconstruct + l_w_embedding * l_embedding + l_w_commitment * l_commitment
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('All Done')

# train pixelcnn
def train_generative_model(vqvae: VQVAE, model, img_shape = None, device = 'cuda', 
                           ckpt_path = None, dataset_type = None, batch_size = 64, n_epochs = 50):
    print('Start training pixelcnn, batch size:', batch_size) # 32
    dataloader = get_dataloader(dataset_type, batch_size, img_shape=img_shape)
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x) # 使用vqvae的encode的输出(这里实际取得是zq，即最近的embedding，而不是ze)，作为pixelcnn的输入

            predict_x = model(x)
            loss = loss_fn(predict_x, x) # cal loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('All Done')

# test the reconstruction of vqvae
def reconstruct(model, x, device, dataset_type):
    print('Start reconstruction ...')
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x) # generate image
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3) # source image and reconstructed image
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'work_dirs/vqvae_reconstruct_{dataset_type}.jpg', x_cat)
    print('All Done')

# use pixelcnn to generate image
def sample_imgs(vqvae: VQVAE, gen_model, img_shape, n_sample = 81, device = 'cuda', dataset_type = None):
    print('Start generation ...') 
    vqvae = vqvae.to(device)
    vqvae.eval()
    gen_model = gen_model.to(device)
    gen_model.eval()

    C, H, W = img_shape
    H, W = vqvae.get_latent_HW((C, H, W))
    input_shape = (n_sample, H, W)
    x = torch.zeros(input_shape).to(device).to(torch.long) # init to zero
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = gen_model(x) # use pixelCNN to generate # B(81) C(32) H(16) W(16)
                prob_dist = F.softmax(output[:, :, i, j], -1) # B C
                pixel = torch.multinomial(prob_dist, 1) # select sample
                x[:, i, j] = pixel[:, 0]

    imgs = vqvae.decode(x) # decode the generate image of pixelcnn
    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(imgs, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1 = int(n_sample**0.5))
    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'work_dirs/vqvae_sample_{dataset_type}.jpg', imgs)

if __name__ == '__main__':
    os.makedirs('./work_dirs', exist_ok = True)
    os.makedirs('./ckpt', exist_ok = True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default = 3) # config
    parser.add_argument('--device', type=int, default = 0) # device
    parser.add_argument('--stage', type=int, default = 0) # stage
    parser.add_argument('--only_test', action = 'store_true') # test
    args = parser.parse_args()
    cfg = get_cfg(args.config) # four training setting

    device = f'cuda:{args.device}'
    img_shape = cfg['img_shape']
    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding']) # VQVAE model
    gen_model = PixelCNNWithEmbedding(cfg['pixelcnn_n_blocks'], # PixelCNN model
                                      cfg['pixelcnn_dim'],
                                      cfg['pixelcnn_linear_dim'], True,
                                      cfg['n_embedding'])
    # stage 1
    if (args.stage == 0):
        if args.only_test == False:
            # 1. Train VQVAE
            train_vqvae(vqvae,
                        img_shape = (img_shape[1], img_shape[2]),
                        device = device,
                        ckpt_path = cfg['vqvae_path'],
                        batch_size = cfg['batch_size'],
                        dataset_type = cfg['dataset_type'],
                        lr = cfg['lr'],
                        n_epochs = cfg['n_epochs'],
                        l_w_embedding = cfg['l_w_embedding'],
                        l_w_commitment = cfg['l_w_commitment']
        )
        else:
            # 2. Test VQVAE by visualizaing reconstruction result
            vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
            dataloader = get_dataloader(cfg['dataset_type'], 16, img_shape = (img_shape[1], img_shape[2]))
            img = next(iter(dataloader)).to(device)
            reconstruct(vqvae, img, device, cfg['dataset_type'])

    # stage 2
    elif (args.stage == 1):
        if args.only_test == False:
            # 3. Train Generative model (Gated PixelCNN in our project)
            vqvae.load_state_dict(torch.load(cfg['vqvae_path'])) # 导入第一阶段训好的VQ-VAE模型
            train_generative_model(vqvae, gen_model, img_shape = (img_shape[1], img_shape[2]), device = device,
                                ckpt_path = cfg['gen_model_path'], dataset_type = cfg['dataset_type'],
                                batch_size = cfg['batch_size_2'], n_epochs = cfg['n_epochs_2'])
        else:
            # 4. Sample VQVAE
            vqvae.load_state_dict(torch.load(cfg['vqvae_path'])) # load VQVAE model
            gen_model.load_state_dict(torch.load(cfg['gen_model_path'])) # load PixelCNN model
            sample_imgs(vqvae, gen_model, cfg['img_shape'], device = device, dataset_type = cfg['dataset_type'])
