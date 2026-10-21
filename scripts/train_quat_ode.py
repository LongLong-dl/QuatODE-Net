import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from dataloader.dataloader import get_quat_ode_loader
from model.quat_ode_net import QuatODENet
from model.loss import NormalizedMAELoss
from model.mambavision_backbone import _load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAMES = ['calories', 'mass', 'fat', 'carb', 'protein']

# 预训练骨干权重
BACKBONE_WEIGHTS = './weights/mambavision_tiny_1k.pth.tar'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def per_target_mae(model, loader):
    model.eval()
    sums = [0.0] * 5
    label_sums = [0.0] * 5
    count = 0

    with torch.no_grad():
        for batch in loader:
            rgb, depth, ingr_ids, mask, targets, _ = batch
            rgb, depth = rgb.to(device), depth.to(device)
            ingr_ids = ingr_ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            pred, _ = model(rgb, depth, ingr_ids, mask,
                            return_intermediates=True)

            err = (pred - targets).abs()
            for k in range(5):
                sums[k] += err[:, k].sum().item()
                label_sums[k] += targets[:, k].abs().sum().item()
            count += targets.size(0)

    model.train()
    results = {}
    total_mae = 0.0
    total_pmae = 0.0
    for i, name in enumerate(TARGET_NAMES):
        mae = sums[i] / count
        pmae = (mae / (label_sums[i] / count)) * 100 if label_sums[i] > 0 else 0.0
        results[name] = {'MAE': round(mae, 2), 'PMAE': round(pmae, 2)}
        total_mae += mae
        total_pmae += pmae

    results['avg_mae'] = round(total_mae / 5, 2)
    results['avg_pmae'] = round(total_pmae / 5, 2)

    return results


def save_attn_analysis(model, loader, log_dir):
    """聚合注意力归因并保存为 attn_analysis.pt。"""
    model.eval()
    attn_sum = None
    delta_abs_sum = None
    count = 0

    with torch.no_grad():
        for batch in loader:
            rgb, depth, ingr_ids, mask, targets, _ = batch
            rgb, depth = rgb.to(device), depth.to(device)
            ingr_ids = ingr_ids.to(device)
            mask = mask.to(device)

            _, inter = model(rgb, depth, ingr_ids, mask,
                             return_intermediates=True)
            attn = inter.get('label_attn')
            delta = inter.get('delta')
            if attn is None or delta is None:
                return None

            attn_sum = attn.sum(dim=0) if attn_sum is None \
                else attn_sum + attn.sum(dim=0)
            delta_abs = delta.abs().sum(dim=0)
            delta_abs_sum = delta_abs if delta_abs_sum is None \
                else delta_abs_sum + delta_abs
            count += attn.shape[0]

    count = max(count, 1)
    attn_mean = attn_sum / count                       # [5, K]
    eps = 1e-12
    p = attn_mean / attn_mean.sum(dim=-1, keepdim=True).clamp(min=eps)
    attn_entropy = -(p * torch.log(p + eps)).sum(dim=-1)   # [5]
    delta_abs_mean = delta_abs_sum / count             # [5]

    kv_names = ['S1', 'S2', 'S3', 'RGB', 'Depth', 'Text']

    payload = {
        'attn_mean': attn_mean.cpu(),
        'attn_entropy': attn_entropy.cpu(),
        'delta_abs_mean': delta_abs_mean.cpu(),
        'label_names': ['cal', 'mass', 'fat', 'carb', 'protein'],
        'kv_names': kv_names,
        'work_dim': 512,
    }
    out_path = os.path.join(log_dir, 'attn_analysis.pt')
    torch.save(payload, out_path)
    print(f'Attention analysis saved to {out_path}')

    model.train()
    return out_path


def main():
    p = argparse.ArgumentParser(
        description='QuatODE-Net Training (无 grams 重构, NG-L4 固定配置)')
    p.add_argument('--backbone_weights', type=str, default=None,
                   help='骨干预训练权重路径（默认 '
                        'weights/mambavision_tiny_1k.pth.tar）')
    p.add_argument('--data_root', type=str,
                   default=os.path.join(_PROJECT_ROOT, 'dataset'))
    p.add_argument('--vocab_path', type=str,
                   default='./dataset/nutrition5k_dataset/imagery/ingredient_vocab.json')
    p.add_argument('--output_dir', type=str, default='runs')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--b', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_backbone', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--early_stop_patience', type=int, default=30)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--exp_tag', type=str, default='final',
                   help='实验 tag：作为 run_name 前缀（<tag>_<timestamp>），并写入 '
                        'config.json / final_summary.json / best_model.pth，便于检索')
    p.add_argument('--text_dim', type=int, default=256)
    p.add_argument('--latent_dim', type=int, default=384)
    p.add_argument('--head_hidden', type=int, default=128)
    p.add_argument('--head_dropout', type=float, default=0.3)
    p.add_argument('--num_ode_steps', type=int, default=4)

    cli = p.parse_args()

    vocab_path = os.path.join(_PROJECT_ROOT, cli.vocab_path) \
        if not os.path.isabs(cli.vocab_path) else cli.vocab_path

    vocab = load_vocab(vocab_path)
    num_ingredients = vocab['num_ingredients']
    print(f'Loaded vocab: {num_ingredients} ingredients')

    set_seed(42)

    model = QuatODENet(
        num_ingredients=num_ingredients,
        text_dim=cli.text_dim,
        latent_dim=cli.latent_dim,
        max_ingredients=20,
        num_ode_steps=cli.num_ode_steps,
        head_hidden=cli.head_hidden,
        head_dropout=cli.head_dropout,
    ).to(device)

    backbone_weights = cli.backbone_weights or BACKBONE_WEIGHTS
    backbone_weights_path = os.path.join(_PROJECT_ROOT, backbone_weights) \
        if not os.path.isabs(backbone_weights) else backbone_weights
    if os.path.isfile(backbone_weights_path):
        print(f'Loading backbone weights from {backbone_weights_path}')
        _load_checkpoint(model.encoder_rgb, backbone_weights_path)
        _load_checkpoint(model.encoder_depth, backbone_weights_path)
    else:
        print(f'Warning: backbone weights not found, using random init')

    print(f'Trainable params: {count_params(model):,}')

    train_loader, test_loader = get_quat_ode_loader(
        data_root=cli.data_root,
        vocab_path=vocab_path,
        size=224,
        batch_size=cli.b,
        num_workers=cli.num_workers,
        max_ingredients=20,
    )

    criterion = NormalizedMAELoss().to(device)

    param_groups = [
        {'params': model.encoder_rgb.parameters(), 'lr': cli.lr_backbone,
         'weight_decay': cli.weight_decay},
        {'params': model.encoder_depth.parameters(), 'lr': cli.lr_backbone,
         'weight_decay': cli.weight_decay},
        {'params': model.text_encoder.parameters(), 'lr': cli.lr,
         'weight_decay': cli.weight_decay},
        {'params': model.fusion.parameters(), 'lr': cli.lr,
         'weight_decay': cli.weight_decay},
    ]
    optimizer = optim.AdamW(param_groups)

    total_epochs = cli.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=cli.lr * 0.025,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cli.exp_tag}_{ts}"
    log_dir = os.path.join(cli.output_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'\nLog directory: {log_dir}')
    print(f'Run name:      {run_name}')
    print(f'Tag:           {cli.exp_tag}\n')

    cfg = {
        "tag": cli.exp_tag,
        "experiment": cli.exp_tag,
        "note": ("QuatODE-Net 无 grams 重构: MambaVision-T + 三模态多尺度 "
                 f"Hamilton ODE + PostFusionLabelRouter, latent_dim={cli.latent_dim}, "
                 f"ode_steps={cli.num_ode_steps}"),
        "num_ingredients": num_ingredients,
        "backbone": "tiny",
        "model": {
            "trainable_params": count_params(model),
            "text_dim": cli.text_dim,
            "latent_dim": cli.latent_dim,
            "head_hidden": cli.head_hidden,
            "head_dropout": cli.head_dropout,
            "num_ode_steps": cli.num_ode_steps,
            "multi_scale": str(model.feature_dims),
            "backbone": "MambaVision-tiny",
            "use_symplectic": True,
            "use_traj_agg": True,
            "traj_agg_learnable": False,
        },
        "training": {
            "epochs": total_epochs, "batch_size": cli.b,
            "loss": criterion.__class__.__name__,
        },
        "optimizer": {"lr": cli.lr, "lr_backbone": cli.lr_backbone,
                      "weight_decay": cli.weight_decay},
    }
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    def test_loss_fn(model, loader, criterion):
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                rgb, depth, ingr_ids, mask, targets, _ = batch
                rgb, depth = rgb.to(device), depth.to(device)
                ingr_ids = ingr_ids.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                pred, _ = model(rgb, depth, ingr_ids, mask,
                                return_intermediates=True)
                total += criterion(pred, targets).item()
                count += 1
        model.train()
        return total / max(count, 1)

    best_mae = float('inf')
    best_epoch = 0
    no_improve_count = 0
    log_path = os.path.join(log_dir, "train_log.jsonl")

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = len(train_loader)

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}'):
            rgb, depth, ingr_ids, mask, targets, _ = batch
            rgb, depth = rgb.to(device), depth.to(device)
            ingr_ids = ingr_ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            nutrition, _ = model(rgb, depth, ingr_ids, mask,
                                 return_intermediates=True)

            loss = criterion(nutrition, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cli.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += (nutrition.detach() - targets).abs().mean().item()

        scheduler.step()
        nb = max(n_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']

        train_loss = epoch_loss / nb
        train_mae = epoch_mae / nb

        test_loss = test_loss_fn(model, test_loader, criterion)
        results = per_target_mae(model, test_loader)
        test_mae = results['avg_mae']

        print(f'Epoch {epoch+1:>3d}/{total_epochs}  '
              f'Train Loss={train_loss:.4f}  '
              f'Train MAE={train_mae:.2f}  '
              f'Test Loss={test_loss:.4f}  '
              f'Test MAE={test_mae:.2f}  '
              f'LR={lr_now:.6f}')

        rec = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_mae": round(train_mae, 2),
            "test_loss": round(test_loss, 4),
            "test_mae": test_mae,
            "test_pmae": results['avg_pmae'],
            "lr": lr_now,
            "timestamp": datetime.now().isoformat(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = epoch + 1
            no_improve_count = 0
            torch.save({
                'tag': cli.exp_tag,
                'run_name': run_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_mae': best_mae,
            }, os.path.join(log_dir, 'best_model.pth'))
            print(f'  >>> New best! Epoch={best_epoch}  Test MAE={best_mae:.2f}')
        else:
            no_improve_count += 1

        if no_improve_count >= cli.early_stop_patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    checkpoint = torch.load(
        os.path.join(log_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'\n=== QuatODE-Net (无 grams) — Per-target MAE on test set ===')
    pt = per_target_mae(model, test_loader)
    for name in TARGET_NAMES:
        print(f'  {name}: MAE={pt[name]["MAE"]}  PMAE={pt[name]["PMAE"]}%')
    print(f'  Average: MAE={pt["avg_mae"]}  PMAE={pt["avg_pmae"]}%')
    print(f'\nBest Test MAE: {best_mae} at epoch {best_epoch}')

    save_attn_analysis(model, test_loader, log_dir)

    with open(os.path.join(log_dir, "final_summary.json"), "w") as f:
        json.dump({
            "tag": cli.exp_tag,
            "best_mae": best_mae, "best_epoch": best_epoch,
            "per_target": {k: v for k, v in pt.items() if k in TARGET_NAMES},
            "avg_mae": pt["avg_mae"], "avg_pmae": pt["avg_pmae"],
        }, f, indent=2)


if __name__ == '__main__':
    main()
