import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from dataloader.dataloader import get_quat_ode_loader
from model.quat_ode_net import QuatODENet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAMES = ['calories', 'mass', 'fat', 'carb', 'protein']


def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_weights_path(path):
    """Resolve weights path: if path is a directory, look for best_model.pth inside."""
    if os.path.isdir(path):
        candidate = os.path.join(path, 'best_model.pth')
        if os.path.isfile(candidate):
            return candidate
        import glob
        candidates = glob.glob(os.path.join(path, '**', 'best_model.pth'), recursive=True)
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            f'best_model.pth not found in directory: {path}')
    return path


def evaluate(weights_path, vocab_path, data_root, num_ingredients=None,
             batch_size=32, num_workers=8, text_dim=256, latent_dim=384,
             head_hidden=128, head_dropout=0.3, num_ode_steps=4):
    """Evaluate final model on test set.

    Args:
        weights_path: path to .pth file, or directory containing best_model.pth
    """
    weights_path = resolve_weights_path(weights_path)

    # 若误传 attn_analysis.pt，回退到同目录的模型权重
    if os.path.basename(weights_path) == 'attn_analysis.pt':
        candidate = os.path.join(
            os.path.dirname(weights_path), 'best_model.pth')
        if os.path.isfile(candidate):
            print(f'[warn] {weights_path} 是注意力分析文件（非模型权重），'
                  f'自动改用 {candidate}')
            weights_path = candidate

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f'Weights file not found: {weights_path}')

    file_size = os.path.getsize(weights_path)
    if file_size == 0:
        raise RuntimeError(f'Weights file is empty (0 bytes): {weights_path}')

    print(f'Weights: {weights_path} ({file_size:,} bytes)')

    vocab_full = load_vocab(vocab_path)
    if num_ingredients is None:
        num_ingredients = vocab_full['num_ingredients']

    model = QuatODENet(
        num_ingredients=num_ingredients,
        text_dim=text_dim,
        latent_dim=latent_dim,
        max_ingredients=20,
        num_ode_steps=num_ode_steps,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    ).to(device)
    model.eval()

    try:
        state = torch.load(weights_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f'Failed to load weights from {weights_path} '
            f'(file may be corrupted or incomplete).\n'
            f'Original error: {e}')

    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'attn_mean' in state:
        raise RuntimeError(
            f'{weights_path} 是 attn_analysis.pt 可解释性文件，不是模型权重；'
            f'请传入同目录的 best_model.pth')

    model.load_state_dict(state)
    print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

    _, test_loader = get_quat_ode_loader(
        data_root=data_root,
        vocab_path=vocab_path,
        size=224,
        batch_size=batch_size,
        num_workers=num_workers,
        max_ingredients=20,
    )

    per_target_abs_sum = [0.0] * 5
    per_target_label_sum = [0.0] * 5
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            rgb, depth, ingr_ids, mask, targets, _ = batch
            rgb, depth = rgb.to(device), depth.to(device)
            ingr_ids = ingr_ids.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            nutrition, _ = model(rgb, depth, ingr_ids, mask,
                                 return_intermediates=True)
            err = (nutrition - targets).abs()

            for k in range(5):
                per_target_abs_sum[k] += err[:, k].sum().item()
                per_target_label_sum[k] += targets[:, k].abs().sum().item()
            count += targets.size(0)

    print(f'\n{"=" * 60}')
    print(f'  QuatODE-Net (无 grams 重构) Evaluation — Test Set ({count} samples)')
    print(f'{"=" * 60}')
    print(f'  {"Target":<12} {"MAE":>8} {"PMAE(%)":>10}')
    print(f'  {"-" * 32}')

    total_mae = 0.0
    results = {}
    for k, name in enumerate(TARGET_NAMES):
        mae = per_target_abs_sum[k] / count
        mean_abs_label = per_target_label_sum[k] / count
        pmae = (mae / mean_abs_label) * 100 if mean_abs_label > 0 else 0.0
        total_mae += mae
        results[name] = {'MAE': round(mae, 2), 'PMAE': round(pmae, 2)}
        print(f'  {name:<12} {mae:>8.2f} {pmae:>9.2f}%')

    avg_mae = round(total_mae / 5, 2)
    avg_pmae = round(sum(r['PMAE'] for r in results.values()) / 5, 2)
    print(f'  {"-" * 32}')
    print(f'  {"AVERAGE":<12} {avg_mae:>8.2f} {avg_pmae:>9.2f}%')
    print(f'{"=" * 60}')

    out_dir = os.path.dirname(weights_path)
    out_path = os.path.join(out_dir, 'eval_quat_ode_result.json')
    eval_data = {
        'weights': weights_path,
        'per_target': results,
        'avg_mae': avg_mae,
        'avg_pmae': avg_pmae,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {out_path}')

    return results, avg_mae, avg_pmae


def main():
    p = argparse.ArgumentParser(
        description='QuatODE-Net Evaluation (无 grams 重构)')
    p.add_argument('--weights', type=str, required=True,
                   help='Path to best_model.pth, or a run directory containing it')
    p.add_argument('--vocab_path', type=str,
                   default='dataset/nutrition5k_dataset/imagery/ingredient_vocab.json')
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--num_ingredients', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--text_dim', type=int, default=256)
    p.add_argument('--latent_dim', type=int, default=384)
    p.add_argument('--head_hidden', type=int, default=128)
    p.add_argument('--head_dropout', type=float, default=0.3)
    p.add_argument('--num_ode_steps', type=int, default=4)

    cli = p.parse_args()

    if cli.data_root is None:
        cli.data_root = os.path.join(_PROJECT_ROOT, 'dataset')
    if not os.path.isabs(cli.vocab_path):
        cli.vocab_path = os.path.join(_PROJECT_ROOT, cli.vocab_path)
    weights_path = os.path.join(_PROJECT_ROOT, cli.weights) \
        if not os.path.isabs(cli.weights) else cli.weights

    evaluate(
        weights_path=weights_path,
        vocab_path=cli.vocab_path,
        data_root=cli.data_root,
        num_ingredients=cli.num_ingredients,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        text_dim=cli.text_dim,
        latent_dim=cli.latent_dim,
        head_hidden=cli.head_hidden,
        head_dropout=cli.head_dropout,
        num_ode_steps=cli.num_ode_steps,
    )


if __name__ == '__main__':
    main()
