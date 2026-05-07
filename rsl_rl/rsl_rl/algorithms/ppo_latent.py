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
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.modules.temporal_gradient_coordinator import TemporalGradientCoordinator
import numpy as np

class PPO_LAT:
    actor_critic: ActorCriticLatent
    discriminator: Discriminator

    def __init__(self,
                 actor_critic,
                 discriminator,
                 wasabi_expert_data,
                 wasabi_state_normalizer,
                 wasabi_style_reward_normalizer,
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

                 discriminator_learning_rate=0.000025,
                 discriminator_momentum=0.9,
                 discriminator_weight_decay=0.0005,
                 discriminator_gradient_penalty_coef=5,
                 discriminator_loss_function="MSELoss", # MSELoss
                 discriminator_num_mini_batches=10,
                 wasabi_replay_buffer_size=100000,

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
        
        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.wasabi_policy_data = ReplayBuffer(discriminator.observation_dim, discriminator.observation_horizon, wasabi_replay_buffer_size, device) #wasabi策略数据
        self.wasabi_expert_data = wasabi_expert_data #wasabi参考数据
        self.wasabi_state_normalizer = wasabi_state_normalizer #wasabi状态规范化
        self.wasabi_style_reward_normalizer = wasabi_style_reward_normalizer #wasabi风格规范化

        # Discriminator parameters
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_momentum = discriminator_momentum
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_loss_function = discriminator_loss_function
        self.discriminator_num_mini_batches = discriminator_num_mini_batches

        if self.discriminator_loss_function == "WassersteinLoss":
            discriminator_optimizer = optim.RMSprop
        else:
            discriminator_optimizer = optim.SGD
        self.discriminator_optimizer = discriminator_optimizer(
                                                    self.discriminator.parameters(),
                                                    lr=self.discriminator_learning_rate,
                                                    momentum=self.discriminator_momentum,
                                                    weight_decay=self.discriminator_weight_decay,
                                                )


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

    def act(self, obs, critic_obs, wasabi_observation_buf, target_velocity=None, swing_command=None):
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
        self.wasabi_observation_buf = wasabi_observation_buf.clone()


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
    
    def process_env_step(self, rewards, dones, infos, wasabi_obs, next_obs=None):
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
        wasabi_observation_buf = torch.cat((self.wasabi_observation_buf[:, 1:], wasabi_obs.unsqueeze(1)), dim=1)
        self.wasabi_policy_data.insert(wasabi_observation_buf)
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
        # wasabi损失
        mean_wasabi_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        
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

                # 时序平滑损失初始化。
                # 当前任务已经加入运动模仿，TemporalGradientCoordinator 只用于 handstand 策略训练：
                # 从四足站立状态逐步过渡到 handstand，不再处理 trot 技能切换。
                temporal_loss = 0.0
                if self.temporal_coef > 0:
                  # 这里只把潜变量 z 作为“状态变化信号”交给时序协调器评估，
                  # 不希望 temporal_loss 反向影响 forward_latent 本身的计算图，因此使用 no_grad。
                  with torch.no_grad():
                    # forward_latent 返回当前 mini-batch 观测对应的潜变量分布参数。
                    # 这里使用均值 z_mean_batch，而不是随机采样的 z，避免采样噪声干扰时序平滑判断。
                    z_mean_batch, _ = self.actor_critic.forward_latent(obs_batch)
                    # tanh 将潜变量压到 [-1, 1]，与 actor 使用潜码时的边界范围保持一致。
                    z_batch = torch.tanh(z_mean_batch)
                if self.prev_z is not None:
                    # 使用当前 batch 潜码与上一 batch 潜码的差值，近似表示“潜空间时序梯度”。
                    # 虽然这里仍传入旧字符串 'transition'，但 TemporalGradientCoordinator 内部会把它映射为
                    # handstand_transition，即“四足站立 -> handstand”的过渡阶段。
                    temporal_loss = self.temporal_coord.compute_temporal_smoothness_loss(
                        # flatten 后计算整体潜码变化方向，便于与历史方向做 dot/cosine 风格比较。
                        z_batch.flatten() - self.prev_z.flatten(),
                        'transition',
                        # 当前潜码和上一批潜码用于计算潜码跳变幅度，防止过渡过程突然切换。
                        z_batch,
                        self.prev_z,
                        # 当前没有向协调器传入速度信息，因此暂时不启用过渡能量骤降惩罚。
                        None,
                    )
                # 保存当前潜码，供下一次 PPO mini-batch 更新时作为时序参考。
                self.prev_z = z_batch.detach()

                # 总损失 = policy + value - entropy bonus + symmetry regularization
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_coef * sym_loss + temporal_loss


                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                if sym_loss:
                    mean_sym_loss += sym_loss.item()

        # Discriminator update
        wasabi_policy_generator = self.wasabi_policy_data.feed_forward_generator(
            self.discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches) #策略数据生成
        wasabi_expert_generator = self.wasabi_expert_data.feed_forward_generator(
            self.discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches) #专家数据生成

        for sample_wasabi_policy, sample_wasabi_expert in zip(wasabi_policy_generator, wasabi_expert_generator):

            # Discriminator loss
            policy_state_buf = torch.zeros_like(sample_wasabi_policy)
            expert_state_buf = torch.zeros_like(sample_wasabi_expert)
            if self.wasabi_state_normalizer is not None:
                for i in range(self.discriminator.observation_horizon):
                    with torch.no_grad():
                        policy_state_buf[:, i] = self.wasabi_state_normalizer.normalize(sample_wasabi_policy[:, i])
                        expert_state_buf[:, i] = self.wasabi_state_normalizer.normalize(sample_wasabi_expert[:, i])
            policy_d = self.discriminator(policy_state_buf.flatten(1, 2))
            expert_d = self.discriminator(expert_state_buf.flatten(1, 2))
            # 判别器损失函数选择
            if self.discriminator_loss_function == "BCEWithLogitsLoss":
                expert_loss = torch.nn.BCEWithLogitsLoss()(expert_d, torch.ones_like(expert_d))
                policy_loss = torch.nn.BCEWithLogitsLoss()(policy_d, torch.zeros_like(policy_d))
            elif self.discriminator_loss_function == "MSELoss":
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            elif self.discriminator_loss_function == "WassersteinLoss":
                expert_loss = -expert_d.mean()
                policy_loss = policy_d.mean()
            else:
                raise ValueError("Unexpected loss function specified")
            wasabi_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(sample_wasabi_expert,
                                                                lambda_=self.discriminator_gradient_penalty_coef) #计算技能判别梯度

            # Gradient step
            #discriminator_loss = wasabi_loss + grad_pen_loss
            discriminator_loss = wasabi_loss
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if self.wasabi_state_normalizer is not None:
                self.wasabi_state_normalizer.update(sample_wasabi_policy[:, 0])
                self.wasabi_state_normalizer.update(sample_wasabi_expert[:, 0])

            mean_wasabi_loss += wasabi_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        policy_num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= policy_num_updates
        mean_surrogate_loss /= policy_num_updates

        discriminator_num_updates = self.discriminator_num_mini_batches
        mean_wasabi_loss /= discriminator_num_updates
        mean_grad_pen_loss /= discriminator_num_updates
        mean_policy_pred /= discriminator_num_updates
        mean_expert_pred /= discriminator_num_updates


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

        return mean_value_loss, mean_surrogate_loss, mean_sym_loss, mean_wasabi_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, temporal_loss, loss 
    
    def _get_skill_labels(self, z):
        """根据潜码值确定技能标签"""
        z_mean = z.mean(dim=-1)
        labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        labels[z_mean < -0.5] = 0      # trot
        labels[(z_mean >= -0.5) & (z_mean < 0.5)] = 1  # transition
        labels[z_mean >= 0.5] = 2      # handstand
        return labels
