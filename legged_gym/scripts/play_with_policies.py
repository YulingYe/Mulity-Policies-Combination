import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
import torch.nn as nn

import numpy as np
import torch

from datetime import datetime

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

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()


    # # load policy

    # train_cfg.runner.resume = True
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    # policy2 = ppo_runner.get_inference_policy(device=env.device)

    
    model1 = Actor(
    num_actor_obs=env_cfg.env.num_single_obs * env_cfg.env.frame_stack,
    num_actions=env_cfg.env.num_actions,
    actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
    init_noise_std=train_cfg.policy.init_noise_std)

    model2 = Actor(
    num_actor_obs=env_cfg.env.num_single_obs * env_cfg.env.frame_stack,
    num_actions=env_cfg.env.num_actions,
    actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
    init_noise_std=train_cfg.policy.init_noise_std)

    model_path1 = '/media/laoye/Localdisk/Ubuntu2004Linux/YuSongmin_go2_gym/logs/go2_trottohandstand1/Aug07_16-02-31_/model_8000.pt'
    model_path2 = '/media/laoye/Localdisk/Ubuntu2004Linux/YuSongmin_go2_gym/logs/go2_trottohandstand2/Aug06_20-28-11_/model_8000.pt'
    model1 = model1.to("cuda:0")
    model2 = model2.to("cuda:0")
    

    try:
        checkpoint = torch.load(model_path1)
        model_state_dict = checkpoint['model_state_dict']
        # 只提取 actor 网络相关的状态字典
        actor_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith('actor')}
        # 加载 std 参数（如果保存了的话）
        if 'std' in model_state_dict:
            actor_state_dict['std'] = model_state_dict['std']
        model1.load_state_dict(actor_state_dict, strict=False)
        print("Actor 模型加载成功！")
    except FileNotFoundError:
        print(f"未找到 {model_path1} 文件，请检查路径。")
    except RuntimeError as e:
        print(f"加载模型时出现错误：{e}，请确保模型结构和保存时一致。")
    except KeyError:
        print(f"检查点文件中缺少 'model_state_dict' 键，请确认保存格式是否正确。")
    
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
        print(f"未找到 {model_path1} 文件，请检查路径。")
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

    

    stop_state_log = 800 # number of steps before plotting states
    model1.eval()
    model2.eval()
    obs = env.get_observations()

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for i in range(10*int(env.max_episode_length)):
        actions1 = model1.actor(obs.detach())
        actions2 = model2.actor(obs.detach())

        if i <= 400:
            
            env_cfg.rewards.cycle_time = 0.5
            env.commands[:, 0]=1.0
            env.commands[:, 1]=0.0
            env.commands[:, 2]=0.
            actions = actions1
            print("+++++-----one ------")
        
        else:
            env_cfg.rewards.cycle_time = 0.5
            env.commands[:, 0]=1.0
            env.commands[:, 1]=0.
            env.commands[:, 2]=0.
            actions = actions2
            print("------two-----")


        # obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
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

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    
    print("args----",args)
    play(args)
