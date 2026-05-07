import torch
from collections import deque


class TemporalGradientCoordinator:
    """
    面向 handstand 策略训练的时序梯度协调器。

    当前任务不再考虑 trot 技能切换，只考虑：
    1. 初始四足站立（stand）
    2. 从 stand 平滑过渡到 handstand（handstand_transition）

    注意：
    - 为了兼容现有 PPO_LAT 调用，外部传入的 skill_type='transition'
      会在内部自动映射为 handstand_transition。
    - 当前模块只负责对潜变量/梯度变化施加平滑约束，不直接建模 trot。
    """

    def __init__(
        self,
        latent_dim,
        window_size=15,
        smooth_lambda=0.3,
        momentum_gain=0.5,
        stand_delta_thresh=0.15,
        transition_delta_thresh=0.30,
        transition_energy_thresh=0.50,
        device='cuda',
    ):
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.smooth_lambda = smooth_lambda
        self.momentum_gain = momentum_gain
        self.stand_delta_thresh = stand_delta_thresh
        self.transition_delta_thresh = transition_delta_thresh
        self.transition_energy_thresh = transition_energy_thresh
        self.device = device

        # 当前仅保留两阶段：
        # stand: 四足稳定站立阶段
        # handstand_transition: 从四足站立向倒立的过渡阶段
        self.gradient_buffer = {
            'stand': deque(maxlen=window_size),
            'handstand_transition': deque(maxlen=window_size),
        }
        self.kinetic_history = {
            'stand': deque(maxlen=window_size),
            'handstand_transition': deque(maxlen=window_size),
        }

    def _canonical_phase(self, skill_type):
        # 兼容旧调用：PPO_LAT 里目前仍传入 'transition'。
        if skill_type in ('transition', 'handstand', 'handstand_transition'):
            return 'handstand_transition'
        return 'stand'

    def _compute_history_direction_loss(self, current_grad, phase):
        history = list(self.gradient_buffer.get(phase, []))
        if len(history) < 2:
            return current_grad.new_tensor(0.0)

        weights = torch.tensor(
            [0.9 ** (len(history) - i) for i in range(len(history))],
            device=current_grad.device,
            dtype=current_grad.dtype,
        )
        weights = weights / weights.sum()
        hist_grads = torch.stack(history)
        hist_mean = (hist_grads * weights.view(-1, 1)).sum(dim=0)

        hist_dir = hist_mean / (hist_mean.norm() + 1e-8)
        current_dir = current_grad / (current_grad.norm() + 1e-8)
        direction_change = 1.0 - torch.dot(current_dir, hist_dir)

        # 只有方向变化过大时才触发平滑惩罚，避免正常更新被过度抑制。
        if direction_change > 0.5:
            return self.smooth_lambda * direction_change
        return current_grad.new_tensor(0.0)

    def _compute_latent_delta_loss(self, current_z, prev_z, phase):
        if prev_z is None:
            return current_z.new_tensor(0.0)

        z_change = torch.norm(current_z - prev_z)
        delta_thresh = (
            self.stand_delta_thresh if phase == 'stand'
            else self.transition_delta_thresh
        )

        # stand 阶段要求潜码变化更稳，transition 阶段允许更大变化。
        if z_change > delta_thresh:
            return 0.1 * (z_change - delta_thresh)
        return current_z.new_tensor(0.0)

    def _compute_transition_energy_loss(self, current_velocity, phase):
        if current_velocity is None:
            return torch.tensor(0.0, device=self.device)

        if phase != 'handstand_transition':
            return current_velocity.new_tensor(0.0)

        kinetic = 0.5 * (current_velocity ** 2)
        kinetic_scalar = kinetic.mean() if kinetic.ndim > 0 else kinetic
        self.kinetic_history[phase].append(kinetic_scalar.detach())

        if len(self.kinetic_history[phase]) < 2:
            return kinetic_scalar.new_tensor(0.0)

        energy_change = kinetic_scalar - self.kinetic_history[phase][-2]

        # 进入 handstand 过渡阶段时，若“能量骤降”，通常意味着潜码切换过猛或动作塌陷。
        if energy_change < 0 and abs(energy_change) > self.transition_energy_thresh:
            return 0.2 * (abs(energy_change) - self.transition_energy_thresh)
        return kinetic_scalar.new_tensor(0.0)

    def compute_temporal_smoothness_loss(
        self,
        current_grad,
        skill_type,
        current_z,
        prev_z=None,
        current_velocity=None,
    ):
        phase = self._canonical_phase(skill_type)

        smooth_loss = current_grad.new_tensor(0.0)
        smooth_loss = smooth_loss + self._compute_history_direction_loss(current_grad, phase)
        smooth_loss = smooth_loss + self._compute_latent_delta_loss(current_z, prev_z, phase)
        smooth_loss = smooth_loss + self._compute_transition_energy_loss(current_velocity, phase)
        return smooth_loss

    def update_buffer(self, skill_type, grad, z):
        phase = self._canonical_phase(skill_type)
        self.gradient_buffer[phase].append(grad.detach().clone())
