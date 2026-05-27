import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path

import torch
import torch.nn as nn

import numpy as np
import torch

from datetime import datetime
from legged_gym.envs.GO2_Mulitpolicy.GO2_JumptoHandstand.GO2_JumptoHandstand_configJ import (
    GO2_JumptoHandstand_Cfg as JumpCfg,
    GO2_JumptoHandstand_PPO as JumpTrainCfg,
)
from legged_gym.envs.GO2_Mulitpolicy.GO2_JumptoHandstand.GO2_JumptoHandstand_configH import (
    GO2_JumptoHandstand_Cfg as HandstandCfg,
    GO2_JumptoHandstand_PPO as HandstandTrainCfg,
)

flag = 0
flag1 = False


def get_activation(activation):
    if activation == 'elu':
        return nn.ELU()
    elif activation == 'relu':
        return nn.ReLU()
    # 可以根据需要添加更多激活函数
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

class Actor(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Actor, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)



        print(f"Actor MLP: {self.actor}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal = torch.distributions.Normal
        Normal.set_default_validate_args = False

def reset_obs(obs):
   
    # 使用切片操作将指定范围的元素赋值为 0
    obs[:, :] = 0
    return obs


def play(args):
    global flag1
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_base_mass = False
    # env_cfg.domain_rand.randomize_base_com = False
    # env_cfg.domain_rand.randomize_pd_gains = False
    # env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.commands.resampling_time=10000000.0
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_towards_goal = False
    env_cfg.domain_rand.add_obs_latency = False
    env_cfg.domain_rand.randomize_obs_motor_latency = False
    env_cfg.domain_rand.randomize_obs_imu_latency = False
    env_cfg.domain_rand.add_cmd_action_latency = False
    env_cfg.domain_rand.randomize_cmd_action_latency = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Jump policy: keep the exact inference path used by play.py, since that branch
    # is already verified to work in the jumptohandstand environment.
    jump_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )

    model2 = Actor(
    num_actor_obs=HandstandCfg.env.num_single_obs * HandstandCfg.env.frame_stack,
    num_actions=HandstandCfg.env.num_actions,
    actor_hidden_dims=HandstandTrainCfg.policy.actor_hidden_dims,
    init_noise_std=HandstandTrainCfg.policy.init_noise_std)

    jump_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', JumpTrainCfg.runner.experiment_name)
    handstand_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', HandstandTrainCfg.runner.experiment_name)
    model_path1 = get_load_path(jump_log_root)
    model_path2 = get_load_path(handstand_log_root)
    model2 = model2.to("cuda:0")
    

    policy1 = None
    try:
        jump_runner.load(model_path1, load_optimizer=False)
        policy1 = jump_runner.get_inference_policy(device=env.device)
        print("Jump policy 加载成功！")
    except FileNotFoundError:
        print(f"未找到 {model_path1} 文件，请检查路径。")
    except RuntimeError as e:
        print(f"加载 jump 模型时出现错误：{e}，请确保模型结构和保存时一致。")
    except KeyError:
        print(f"jump 检查点文件中缺少 'model_state_dict' 键，请确认保存格式是否正确。")

    if policy1 is None:
        raise RuntimeError(f"Jump policy 未成功加载，无法继续执行: {model_path1}")
    
    try:
        checkpoint = torch.load(model_path2)
        model_state_dict = checkpoint['model_state_dict']
        # 只提取 actor 网络相关的状态字典
        actor_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith('actor')}
        # 加载 std 参数（如果保存了的话）
        if 'std' in model_state_dict:
            actor_state_dict['std'] = model_state_dict['std']
        model2.load_state_dict(actor_state_dict, strict=False)
        print("Actor 模型加载成功！")
    except FileNotFoundError:
        print(f"未找到 {model_path2} 文件，请检查路径。")
    except RuntimeError as e:
        print(f"加载模型时出现错误：{e}，请确保模型结构和保存时一致。")
    except KeyError:
        print(f"检查点文件中缺少 'model_state_dict' 键，请确认保存格式是否正确。")

    
    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 400 # number of steps before plotting states
    stop_rew_log = 800 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    env.commands[:, 0]=1.0
    env.commands[:, 1]=0.
    env.commands[:, 2]=0.
    # for i in range(10*int(env.max_episode_length)):
    #     if i <= 100:
    #         actions = policy(obs.detach())
    #         obs, _, rews, dones, infos = env.step(actions.detach())
    #         # print(obs[0,3:6])
    #         if RECORD_FRAMES:
    #                     if i % 2:
    #                         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
    #                         env.gym.write_viewer_image_to_file(env.viewer, filename)
    #                         img_idx += 1 
    #         if MOVE_CAMERA:
    #             camera_position += camera_vel * env.dt
    #             env.set_camera(camera_position, camera_position + camera_direction)
        
    #     else:
    #         actions = policy2(obs.detach())
    #         print("-----------------here--------------------++++++++")

    #     if i < stop_state_log:
    #         logger.log_states(
    #             {
    #                 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
    #                 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
    #                 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
    #                 'dof_torque': env.torques[robot_index, joint_index].item(),
    #                 'command_x': env.commands[robot_index, 0].item(),
    #                 'command_y': env.commands[robot_index, 1].item(),
    #                 'command_yaw': env.commands[robot_index, 2].item(),
    #                 'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
    #                 'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
    #                 'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
    #                 'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
    #                 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
    #             }
    #         )
    #     elif i==stop_state_log:
    #         logger.plot_states()
    #     if  0 < i < stop_rew_log:
    #         if infos["episode"]:
    #             num_episodes = torch.sum(env.reset_buf).item()
    #             if num_episodes>0:
    #                 logger.log_rewards(infos["episode"], num_episodes)
    #     elif i==stop_rew_log:
    #         logger.print_rewards()

    

    stop_state_log = 470 # number of steps before plotting states #800
#    model1.eval()
    model2.eval()
    jump_cycle_time = JumpCfg.rewards.cycle_time
    handstand_cycle_time = HandstandCfg.rewards.cycle_time
    jump_p_gain = JumpCfg.control.stiffness['joint']
    jump_d_gain = JumpCfg.control.damping['joint']
    handstand_p_gain = HandstandCfg.control.stiffness['joint']
    handstand_d_gain = HandstandCfg.control.damping['joint']
    switch_step = 400
    in_stand_mode = False

    env.set_policy_mode(
        stand_mode=False,
        p_gain=jump_p_gain,
        d_gain=jump_d_gain,
        cycle_time=jump_cycle_time,
        reset_history=True,
    )
    obs = env.get_policy_observation(stand_mode=False)

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for i in range(10*int(env.max_episode_length)):
        if i <= switch_step:
            env.commands[:, 0] = 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            actions = policy1(obs.detach())
            print("+++++-----one ------")
        else:
            if not in_stand_mode:
                env.set_policy_mode(
                    stand_mode=True,
                    p_gain=handstand_p_gain,
                    d_gain=handstand_d_gain,
                    cycle_time=handstand_cycle_time,
                    reset_history=True,
                )
                obs = env.get_policy_observation(stand_mode=True)
                in_stand_mode = True
            env.commands[:, 0] = 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            actions = model2.actor(obs.detach())
            print("------two-----")


        # obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        if i> 350 and i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + 0.5,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        
        obs, _, rews, dones, infos = env.step(actions.detach())
        if in_stand_mode:
            obs = env.get_policy_observation(stand_mode=True)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    
    print("args----",args)
    play(args)
