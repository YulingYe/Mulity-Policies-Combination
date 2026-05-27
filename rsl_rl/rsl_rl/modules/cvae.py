import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEncoder(nn.Module):
    """条件编码器 q(z | x, c)"""
    def __init__(self, obs_dim, cond_dim, latent_dim, hidden_dims=[256, 256]):
        super().__init__()
        # 输入：观测 x + 条件 c
        input_dim = obs_dim + cond_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x, c):
        # x: [batch, obs_dim], c: [batch, cond_dim]
        h = torch.cat([x, c], dim=-1)
        h = self.shared(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class ConditionalDecoder(nn.Module):
    """条件解码器 p(x | z, c)"""
    def __init__(self, latent_dim, cond_dim, obs_dim, hidden_dims=[256, 256]):
        super().__init__()
        input_dim = latent_dim + cond_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        # z: [batch, latent_dim], c: [batch, cond_dim]
        h = torch.cat([z, c], dim=-1)
        return self.net(h)

class CVAE(nn.Module):
    """
    条件变分自编码器
    用于在给定条件 c 下学习观测 x 的潜在表示 z
    """
    def __init__(self, obs_dim, cond_dim, latent_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.obs_dim = obs_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.encoder = ConditionalEncoder(obs_dim, cond_dim, latent_dim, hidden_dims)
        self.decoder = ConditionalDecoder(latent_dim, cond_dim, obs_dim, hidden_dims)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, c):
        mu, logvar = self.encoder(x, c)
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std

    def forward(self, x, c):
        mu, logvar, _ = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar, z

    def loss(self, x, c, beta=1.0):
        """
        计算 CVAE 的 ELBO 损失（负值）
        x: [batch, obs_dim]
        c: [batch, cond_dim]
        """
        x_recon, mu, logvar, _ = self.forward(x, c)
        # 重建损失：高斯似然（MSE）
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        # KL 散度：N(mu, var) vs N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def sample_latent(self, x, c, deterministic=False):
        mu, logvar, std = self.encode(x, c)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar, std

    def sample(self, c, num_samples=None):
        """生成给定条件 c 下的新样本"""
        if num_samples is not None:
            c = c.repeat(num_samples, 1)
        batch = c.size(0)
        z = torch.randn(batch, self.latent_dim, device=c.device)
        x_gen = self.decoder(z, c)
        return x_gen
