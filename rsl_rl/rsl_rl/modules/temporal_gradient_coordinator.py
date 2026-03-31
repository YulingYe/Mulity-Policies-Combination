# rsl_rl/modules/temporal_gradient_coordinator.py
import torch
from collections import deque

class TemporalGradientCoordinator:
    def __init__(self, latent_dim, window_size=15, smooth_lambda=0.3, momentum_gain=0.5, device='cuda'):
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.smooth_lambda = smooth_lambda
        self.momentum_gain = momentum_gain
        self.device = device
        self.gradient_buffer = {
            'trot': deque(maxlen=window_size),
            'transition': deque(maxlen=window_size),
            'handstand': deque(maxlen=window_size)
        }
        self.kinetic_history = deque(maxlen=window_size)

    def compute_temporal_smoothness_loss(self, current_grad, skill_type, current_z, prev_z=None, current_velocity=None):
        smooth_loss = 0.0
        # 1. 梯度方向平滑
        history = list(self.gradient_buffer.get(skill_type, []))
        if len(history) >= 2:
            weights = torch.tensor([0.9**(len(history)-i) for i in range(len(history))], device=self.device)
            weights = weights / weights.sum()
            hist_grads = torch.stack(history)
            hist_mean = (hist_grads * weights.view(-1, 1)).sum(dim=0)
            hist_dir = hist_mean / (hist_mean.norm() + 1e-8)
            current_dir = current_grad / (current_grad.norm() + 1e-8)
            direction_change = 1 - torch.dot(current_dir, hist_dir)
            if direction_change > 0.5:
                smooth_loss += self.smooth_lambda * direction_change
        # 2. 潜码平滑
        if prev_z is not None:
            z_change = torch.norm(current_z - prev_z)
            if z_change > 0.3:
                smooth_loss += 0.1 * z_change
        # 3. 动能耗散感知
        if current_velocity is not None and skill_type == 'transition':
            kinetic = 0.5 * (current_velocity ** 2)
            self.kinetic_history.append(kinetic.item())
            if len(self.kinetic_history) >= 2:
                energy_change = kinetic - self.kinetic_history[-2]
                if energy_change < 0 and abs(energy_change) > 0.5:
                    smooth_loss += 0.2 * abs(energy_change)
        return smooth_loss

    def update_buffer(self, skill_type, grad, z):
        self.gradient_buffer[skill_type].append(grad.detach().clone())