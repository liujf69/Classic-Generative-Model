import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_features, latent_size, y_size=0):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder_forward = nn.Sequential( # encoder
            nn.Linear(in_features + y_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, self.latent_size * 2)
        )

        self.decoder_forward = nn.Sequential( # decoder
            nn.Linear(self.latent_size + y_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )

    def encoder(self, X): # encode
        out = self.encoder_forward(X) # 这里通过一个encoder生成均值和标准差
        mu = out[:, :self.latent_size] # 输出的前半部分作为均值
        log_var = out[:, self.latent_size:] # 后半部分作为标准差
        return mu, log_var

    def decoder(self, z): # decode
        mu_prime = self.decoder_forward(z)
        return mu_prime

    def reparameterization(self, mu, log_var): # reparameterization
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var): # cal loss
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))
        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))
        return reconstruction_loss + latent_loss

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X) # encode
        z = self.reparameterization(mu, log_var) # generate z by reparameterization
        mu_prime = self.decoder(z) # decode
        return mu_prime, mu, log_var

class CVAE(VAE):
    def __init__(self, in_features, latent_size, y_size):
        super(CVAE, self).__init__(in_features, latent_size, y_size)

    def forward(self, X, y=None, *args, **kwargs):
        y = y.to(next(self.parameters()).device)
        X_given_Y = torch.cat((X, y.unsqueeze(1)), dim=1)

        mu, log_var = self.encoder(X_given_Y)
        z = self.reparameterization(mu, log_var)
        z_given_Y = torch.cat((z, y.unsqueeze(1)), dim=1)

        mu_prime_given_Y = self.decoder(z_given_Y)
        return mu_prime_given_Y, mu, log_var