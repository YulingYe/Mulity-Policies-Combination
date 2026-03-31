import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

class SwingUpGradientCoordinator:
    """
    起摆动力学感知的梯度协调器
    解决trot→handstand切换中的动量控制问题
    参考：MPC无参考框架的起摆控制[citation:1] + 腾讯Max起摆算法[citation:6]
    """
    def __init__(self, window_size=15, smooth_lambda=0.3, momentum_gain=0.5):
        self.gradient_buffer = {'trot': [], 'handstand': []}
        self.window_size = window_size
        self.smooth_lambda = smooth_lambda
        self.momentum_gain = momentum_gain
        self.prev_swing_phase = None
        
    def compute_swing_up_loss(self, current_grad, skill_type, 
                               current_velocity, base_orientation, swing_phase):
        """
        计算起摆过程损失
        关键：在起摆瞬间施加动量引导梯度
        """
        # 1. 基础时序平滑约束
        history = self.gradient_buffer.get(skill_type, [])
        temporal_loss = self._temporal_smoothness_loss(current_grad, history)
        
        # 2. 起摆动量引导损失
        swing_up_loss = 0.0
        
        # 检测起摆触发条件（速度降低到阈值，且身体开始前倾）
        if current_velocity < 0.8 and base_orientation.pitch > 0.3:
            # 起摆阶段：鼓励前腿离地、身体直立
            # 梯度向handstand方向引导
            swing_up_loss = -self.momentum_gain * current_grad.dot(self.handstand_direction)
        
        # 3. 双足平衡约束损失
        # 在handstand阶段，惩罚前腿触地
        if swing_phase == 'handstand':
            front_contact_loss = self._penalize_front_contact()
        else:
            front_contact_loss = 0.0
        
        self.gradient_buffer[skill_type] = (
            [current_grad] + history
        )[:self.window_size]
        
        return temporal_loss + 0.5 * swing_up_loss + 0.3 * front_contact_loss