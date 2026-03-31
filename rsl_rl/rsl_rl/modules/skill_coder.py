import torch
import torch.nn as nn
import torch.nn.functional as F

class SkillEncoder(nn.Module):
    """
    ASE风格技能编码器
    将状态映射到潜空间，同时支持互信息最大化
    参考: LocomotionWithNP3O项目[citation:1]
    """
    def __init__(self, state_dim, latent_dim=16, history_len=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_len = history_len
        
        # 历史编码器
        self.history_encoder = nn.Sequential(
            nn.Linear(state_dim * history_len, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim * 2)  # mean + log_std
        )
        
        # 对比学习投影头（Barlow Twins风格[citation:1]）
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, history):
        """
        将状态历史编码为潜码分布
        history: [batch, history_len, state_dim]
        """
        batch_size = history.shape[0]
        history_flat = history.view(batch_size, -1)
        
        output = self.history_encoder(history_flat)
        mean, log_std = output.chunk(2, dim=-1)
        std = log_std.exp().clamp(0.01, 1.0)
        
        # 重参数化采样
        z = mean + std * torch.randn_like(mean)
        z = torch.tanh(z)  # 约束到[-1, 1]
        
        return z, mean, std
    
    def compute_contrastive_loss(self, z_t, z_t1):
        """
        计算对比学习损失
        拉近相邻时间步的潜码（时序平滑约束）
        参考: LocomotionWithNP3O的对比学习设计[citation:1]
        """
        # 投影到对比空间
        p_t = self.projection_head(z_t)
        p_t1 = self.projection_head(z_t1)
        
        # 余弦相似度
        p_t = F.normalize(p_t, dim=-1)
        p_t1 = F.normalize(p_t1, dim=-1)
        
        # 正样本：同一轨迹相邻时间步
        pos_sim = (p_t * p_t1).sum(dim=-1)
        
        # 负样本：不同轨迹之间
        neg_mask = ~torch.eye(z_t.shape[0], dtype=torch.bool, device=z_t.device)
        neg_sim = torch.mm(p_t, p_t1.T)[neg_mask].view(z_t.shape[0], -1)
        
        # InfoNCE损失
        temperature = 0.07
        pos_exp = torch.exp(pos_sim / temperature)
        neg_exp = torch.exp(neg_sim / temperature).sum(dim=-1)
        
        contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
        
        return contrastive_loss


class SkillDecoder(nn.Module):
    """
    技能解码器：潜码 + 状态 → 动作
    """
    def __init__(self, state_dim, latent_dim, action_dim, hidden_dims=[256, 128]):
        super().__init__()
        
        layers = []
        input_dim = state_dim + latent_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h),
                nn.ELU(),
                nn.LayerNorm(h)
            ])
            input_dim = h
        
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, state, z):
        """解码动作"""
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)