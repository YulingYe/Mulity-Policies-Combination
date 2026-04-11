import torch
import torch.nn as nn
import torch.nn.functional as F

class CASSIDiscriminator(nn.Module):
    """
    CASSI合作对抗判别器
    区分技能类型 (trot, transition, handstand)
    """
    def __init__(self, state_dim, action_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        # 判别器输入：s, a, s', z
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 classes
            nn.LogSoftmax(dim=-1)
        )
        # 合作性引导网络
        self.cooperative_guide = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action, next_state, z):
        x = torch.cat([state, action, next_state, z], dim=-1)
        return self.discriminator(x)

    def compute_cooperative_signal(self, z):
        return self.cooperative_guide(z)

    def cassi_loss(self, state, action, next_state, z, skill_labels):
        logits = self.forward(state, action, next_state, z)
        adv_loss = F.nll_loss(logits, skill_labels)
        coop_signal = self.compute_cooperative_signal(z)
        coop_loss = -torch.log(coop_signal + 1e-8).mean()
        return adv_loss + 0.1 * coop_loss