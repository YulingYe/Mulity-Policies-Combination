# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticLatent
from rsl_rl.storage import RolloutStorage
from rsl_rl.modules import discriminator_ensemble
from rsl_rl.modules.skill_coder import CASSIDiscriminator
from rsl_rl.modules.temporal_gradient_coordinator import TemporalGradientCoordinator
import numpy as np

class PPO_LAT:
    actor_critic: ActorCriticLatent
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 sym_loss = False,
                 obs_permutation = None,
                 act_permutation = None,
                 frame_stack = 15,
                 sym_coef = 1.0,
                 # 新增参数
                 latent_dim=16,
                 cassi_coef=0.1,
                 temporal_coef=0.05,
                 swing_trigger_vel=0.8,
                 full_stand_vel=0.3
                 ):
        # PPO_LAT 在标准 PPO 基础上额外支持：
        # 1. 对称性约束（symmetry loss）
        # 2. 一段实验性的 latent / 起摆切换逻辑
        # 这里先完成训练器本身的状态初始化。

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.sym_loss = sym_loss
        self.sym_coef = sym_coef
        if self.sym_loss:
            # act_perm_mat / obs_perm_mat 用置换矩阵编码“镜像映射”：
            # 给定原始动作或观测，可通过矩阵乘法得到左右对称后的版本。
            self.act_perm_mat = torch.zeros((len(act_permutation), len(act_permutation))).cuda()
            for i, perm in enumerate(act_permutation):
                self.act_perm_mat[int(abs(perm))][i] = np.sign(perm) 
            obs_permutation_stack = []
            for i in range(frame_stack):
                for p in obs_permutation:
                    obs_permutation_stack.append(np.sign(p)*(abs(p)+i*len(obs_permutation)))  
            self.obs_perm_mat = torch.zeros((len(obs_permutation_stack), len(obs_permutation_stack))).cuda()
            for i, perm in enumerate(obs_permutation_stack):
                self.obs_perm_mat[int(abs(perm))][i] = np.sign(perm)  

        # 新增组件
        self.latent_dim = latent_dim
        self.cassi_coef = cassi_coef
        self.temporal_coef = temporal_coef
        self.swing_trigger_vel = swing_trigger_vel
        self.full_stand_vel = full_stand_vel

        # # CASSI判别器
        # self.cassi_disc = CASSIDiscriminator(
        #     state_dim=self.actor_critic.num_actor_obs,
        #     action_dim=self.actor_critic.num_actions,
        #     latent_dim=latent_dim
        # ).to(device)
        # self.cassi_optimizer = optim.Adam(self.cassi_disc.parameters(), lr=1e-4)

        # 时序协调器
        self.temporal_coord = TemporalGradientCoordinator(
            latent_dim=latent_dim, device=device
        )
        self.prev_z = None  # 存储上一步潜码

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # rollout buffer 在拿到环境规模后再创建，避免在 __init__ 中依赖任务配置。
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, target_velocity=None, swing_command=None):
        # 采样阶段：
        # 1. 用 actor 生成动作
        # 2. 用 critic 估计 value
        # 3. 把当前 step 所需信息缓存到 self.transition，供 env.step() 后写入 rollout
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # 潜码生成
        z, z_log_prob, z_mean, z_std = self.actor_critic.get_latent(obs, deterministic=False)

        # 边界调制（根据速度和命令）
        if target_velocity is not None:
            # 获取平均速度
            vel_mean = target_velocity.mean().item()
            # 根据速度调整潜码（例如，低速时强制向handstand移动）
            if swing_command is not None and swing_command > 0.5:
                z = torch.clamp(z, 0.2, 1.0)
            elif vel_mean < self.full_stand_vel:
                z = torch.clamp(z, 0.5, 1.0)   # 完全站立区域
            elif vel_mean < self.swing_trigger_vel:
                # 过渡区，不作强制，允许混合
                pass
        else:
            vel_mean = None

        # need to record obs and critic_obs before env.step()
     
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs


        # 扩展存储潜码相关信息
        self.transition.z = z.detach()
        self.transition.z_log_prob = z_log_prob.detach()
        self.transition.z_mean = z_mean.detach()
        self.transition.z_std = z_std.detach()
        self.transition.target_velocity = target_velocity if target_velocity is not None else torch.zeros(obs.shape[0], device=self.device)
        self.transition.swing_command = swing_command if swing_command is not None else torch.zeros(obs.shape[0], device=self.device)
                
       # 记录CASSI所需数据（会在process_env_step中使用下一状态）
        self._cassi_state = obs.clone()
        self._cassi_action = self.transition.actions.clone()
        self._cassi_z = z.clone()

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_obs=None):
        # env.step() 返回后，把奖励/终止标记补到当前 transition 上并写入 buffer。
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            # 对 time out 终止做 bootstrap，避免把时间截断误当作真实终止。
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        
        # 存储CASSI数据
        # self.transition.cassi_state = self._cassi_state
        # self.transition.cassi_action = self._cassi_action
        # self.transition.cassi_next_state = next_obs.clone()
        # self.transition.cassi_z = self._cassi_z


        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)


    
    def compute_returns(self, last_critic_obs):
        
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        # 对 rollout 做多轮 epoch / mini-batch PPO 更新。
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_sym_loss = 0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                #sym loss
                sym_loss = 0 
                if self.sym_loss:
                    # 对称性损失的目标：
                    # “镜像后的观测输入 actor，再把输出动作镜像回原空间”
                    # 应尽量与当前 batch 的动作均值一致。
                    # 这样能鼓励策略满足左右对称先验。
                    mirror_obs = torch.matmul(obs_batch,self.obs_perm_mat)#对称的观察
                    mirror_act = self.actor_critic.actor(mirror_obs)#对称的观察的输出的对陈的动作
                    m_mirror_act = torch.matmul(mirror_act,self.act_perm_mat)#将对称的观察的动作映射到原动作空间
                    sym_loss = (mu_batch-m_mirror_act).pow(2).mean()
                    # print("shapes:",obs_batch.shape,mirror_obs.shape,mirror_act.shape,m_mirror_act.shape,mu_batch.shape)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        # 用当前策略和旧策略的高斯参数近似 KL，
                        # 若偏离过大则降学习率，偏离过小则升学习率。
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate



                # Surrogate loss
                # PPO 核心目标：importance ratio 与 clip 后版本取较大者。
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    # value clip 与 policy clip 同理，限制 critic 单次更新步长。
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                temporal_loss = 0.0
                if self.temporal_coef > 0:
                  with torch.no_grad():
                    z_mean_batch, _ = self.actor_critic.forward_latent(obs_batch)
                    z_batch = torch.tanh(z_mean_batch)
                if self.prev_z is not None:
                    temporal_loss = self.temporal_coord.compute_temporal_smoothness_loss(
                        z_batch.flatten() - self.prev_z.flatten(),
                        'transition',
                        z_batch,
                        self.prev_z,
                        None,
                    )
                self.prev_z = z_batch.detach()

                # 总损失 = policy + value - entropy bonus + symmetry regularization
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_coef * sym_loss


                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if sym_loss:
                    mean_sym_loss += sym_loss.item()


        # # CASSI损失
        # cassi_loss = 0
        # if cassi_state_batch is not None:
        #         # 获取技能标签（基于潜码）
        #         with torch.no_grad():
        #             skill_labels = self._get_skill_labels(cassi_z_batch)
        #         cassi_loss = self.cassi_disc.cassi_loss(
        #             cassi_state_batch, cassi_action_batch, cassi_next_state_batch,
        #             cassi_z_batch, skill_labels
        #         )

        # 时序平滑项当前只保留轻量状态统计，不再额外构建/保留第二份反向图。
        # 原来的 torch.autograd.grad(..., retain_graph=True) 会在 PPO 反传之后再次追踪整张图，
        # 显存开销很大，而且这部分损失目前并没有并入最终优化目标。
        # temporal_loss = 0.0
        # if self.temporal_coef > 0:
        #     with torch.no_grad():
        #         z_mean_batch, _ = self.actor_critic.forward_latent(obs_batch)
        #         z_batch = torch.tanh(z_mean_batch)
        #         if self.prev_z is not None:
        #             temporal_loss = self.temporal_coord.compute_temporal_smoothness_loss(
        #                 z_batch.flatten() - self.prev_z.flatten(),
        #                 'transition',
        #                 z_batch,
        #                 self.prev_z,
        #                 None,
        #             )
        #         self.prev_z = z_batch.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        if sym_loss:
            mean_sym_loss /= num_updates
        else:
            mean_sym_loss =0
        # 一个 rollout 的数据只使用一次，更新完成后清空缓存。
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_sym_loss, temporal_loss, loss 
    
    def _get_skill_labels(self, z):
        """根据潜码值确定技能标签"""
        z_mean = z.mean(dim=-1)
        labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        labels[z_mean < -0.5] = 0      # trot
        labels[(z_mean >= -0.5) & (z_mean < 0.5)] = 1  # transition
        labels[z_mean >= 0.5] = 2      # handstand
        return labels
