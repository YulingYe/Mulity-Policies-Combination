#!/usr/bin/env python3
"""
pretrain_cvae.py

预训练 CVAE 模型，利用参考轨迹数据（例如 trot 步态）。
用法示例：
    python pretrain_cvae.py --ref_path reference_trot.pt --save_path cvae_trot_pretrained.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 导入你的 CVAE 模块（确保 cvae.py 在 Python 路径中）
from rsl_rl.modules.cvae import CVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain CVAE on reference trajectories")
    parser.add_argument("--ref_path", type=str, required=True, help="Path to reference .pt file (shape: [num_traj, traj_len, obs_dim])")
    parser.add_argument("--save_path", type=str, default="cvae_pretrained.pt", help="Path to save the trained CVAE model")
    parser.add_argument("--obs_dim", type=int, default=41, help="Observation dimension (from reference data)")
    parser.add_argument("--cond_dim", type=int, default=8, help="Condition dimension")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256], help="Hidden layer dimensions for encoder/decoder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for KL term in ELBO (beta-VAE)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--skill_label", type=int, default=0, help="Skill label for this reference data (used to build one-hot condition)")
    parser.add_argument("--num_skills", type=int, default=3, help="Total number of skills (trot, handstand, jump, ...)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. 加载参考数据
    print(f"Loading reference data from {args.ref_path}...")
    data = torch.load(args.ref_path, map_location="cpu")
    if data.dim() == 3:
        num_traj, traj_len, obs_dim = data.shape
        assert obs_dim == args.obs_dim, f"Data obs_dim {obs_dim} != specified {args.obs_dim}"
        flat_obs = data.view(-1, obs_dim).to(device)   # [N, obs_dim]
        print(f"Loaded {num_traj} trajectories, each {traj_len} steps -> total {flat_obs.shape[0]} samples")
    else:
        raise ValueError(f"Expected 3D tensor [num_traj, traj_len, obs_dim], got shape {data.shape}")

    # 2. 构造条件 c (one-hot 编码技能标签)
    # 假设所有样本都属于同一个技能（由 --skill_label 指定）
    total_samples = flat_obs.shape[0]
    cond = torch.zeros(total_samples, args.cond_dim, device=device)
    # 如果 cond_dim >= num_skills，使用 one‑hot；否则使用可学习的嵌入（这里简化）
    if args.cond_dim >= args.num_skills:
        cond[:, args.skill_label] = 1.0
    else:
        # cond_dim 小于技能数，用 skill_label 模 cond_dim 作为索引
        idx = args.skill_label % args.cond_dim
        cond[:, idx] = 1.0
    print(f"Condition shape: {cond.shape}, using skill label {args.skill_label}")

    # 3. 创建 DataLoader
    dataset = TensorDataset(flat_obs, cond)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 4. 构建 CVAE 模型
    cvae = CVAE(
        obs_dim=args.obs_dim,
        cond_dim=args.cond_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    ).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)

    # 5. 训练循环
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        for x_batch, c_batch in dataloader:
            optimizer.zero_grad()
            loss, recon, kl = cvae.loss(x_batch, c_batch, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | KL: {avg_kl:.6f}")

    # 6. 保存模型
    torch.save({
        'model_state_dict': cvae.state_dict(),
        'obs_dim': args.obs_dim,
        'cond_dim': args.cond_dim,
        'latent_dim': args.latent_dim,
        'hidden_dims': args.hidden_dims,
        'beta': args.beta,
    }, args.save_path)
    print(f"Model saved to {args.save_path}")

    # 可选：测试生成样本
    with torch.no_grad():
        sample_cond = cond[:args.batch_size]
        generated_obs = cvae.sample(sample_cond)
        print(f"Sample generation test: generated obs shape {generated_obs.shape}")

if __name__ == "__main__":
    main()