"""Nutrition5K DataLoader：RGB-D 数据集与 QuatODE-Net 专用 loader。"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader.dataset import Nutrition_RGBD_CSV, Nutrition_RGBD_Ingredient


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_DataLoader(args):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((args.size + 50, args.size + 100)),
        transforms.CenterCrop((args.size, args.size)),
        transforms.ToTensor(),
        norm,
    ])

    data_root = os.path.join(args.data_root, 'nutrition5k_dataset') if \
        os.path.basename(args.data_root) != 'nutrition5k_dataset' else args.data_root

    imagery_root = os.path.join(data_root, 'imagery')
    train_csv = os.path.join(imagery_root, 'train.csv')
    test_csv = os.path.join(imagery_root, 'test.csv')

    if getattr(args, 'use_ingredient', False):
        vocab_path = getattr(args, 'vocab_path',
                             os.path.join(data_root, 'imagery', 'ingredient_vocab.json'))
        max_ingredients = getattr(args, 'max_ingredients', 20)
        index_path = getattr(args, 'index_path', None)
        return_grams = getattr(args, 'return_grams', False)
        trainset = Nutrition_RGBD_Ingredient(
            data_root, train_csv, vocab_path,
            transform=transform, max_ingredients=max_ingredients,
            index_path=index_path, return_grams=return_grams,
        )
        testset = Nutrition_RGBD_Ingredient(
            data_root, test_csv, vocab_path,
            transform=transform, max_ingredients=max_ingredients,
            index_path=index_path, return_grams=return_grams,
        )
    else:
        trainset = Nutrition_RGBD_CSV(imagery_root, train_csv, transform=transform)
        testset = Nutrition_RGBD_CSV(imagery_root, test_csv, transform=transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              drop_last=True)

    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=seed_worker)

    return train_loader, test_loader


def get_quat_ode_loader(data_root, vocab_path, size=224, batch_size=32,
                        num_workers=8, max_ingredients=20,
                        index_path=None, return_grams=False):
    """QuatODE-Net 专用 DataLoader，固定使用含食材的 RGB-D 数据集。

    Args:
        data_root:       nutrition5k_dataset 根目录
        vocab_path:      ingredient_vocab.json 路径
        size:            图像尺寸
        batch_size:      batch size
        num_workers:     DataLoader workers
        max_ingredients: 最大食材数
        index_path:      预处理索引路径（缺失时回退 CSV）
        return_grams:    True 时返回含 grams 的 7 元组

    Returns:
        train_loader, test_loader

    Batch format (6 元组):
        rgb [B,3,H,W], depth [B,3,H,W], ingr_ids [B,N] (0=pad, 1=unk),
        mask [B,N], targets [B,5], dish_ids
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((size + 50, size + 100)),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        norm,
    ])

    data_root = os.path.join(data_root, 'nutrition5k_dataset') if \
        os.path.basename(data_root) != 'nutrition5k_dataset' else data_root

    imagery_root = os.path.join(data_root, 'imagery')
    train_csv = os.path.join(imagery_root, 'train.csv')
    test_csv = os.path.join(imagery_root, 'test.csv')

    trainset = Nutrition_RGBD_Ingredient(
        data_root, train_csv, vocab_path,
        transform=transform, max_ingredients=max_ingredients,
        index_path=index_path, return_grams=return_grams,
    )
    testset = Nutrition_RGBD_Ingredient(
        data_root, test_csv, vocab_path,
        transform=transform, max_ingredients=max_ingredients,
        index_path=index_path, return_grams=return_grams,
    )

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              drop_last=True)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             worker_init_fn=seed_worker)

    print(f'QuatODE loader: train={len(train_loader)} batches, '
          f'test={len(test_loader)} batches, '
          f'ingredients={trainset.num_ingredients}')

    return train_loader, test_loader


if __name__ == '__main__':
    class Args:
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'dataset')
        size = 224
        b = 4
        num_workers = 0
        use_ingredient = True
        vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'data', 'ingredient_vocab.json')
        max_ingredients = 20
        index_path = None
        return_grams = False

    train_loader, test_loader = get_DataLoader(Args)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    img_rgb, img_depth, ingr_ids, mask, targets, dish_ids = batch
    print(f"RGB:      {img_rgb.shape}")
    print(f"Depth:    {img_depth.shape}")
    print(f"Ingr IDs: {ingr_ids.shape}")
    print(f"Mask:     {mask.shape}")
    print(f"Targets:  {targets.shape}")

    print("\n=== All DataLoader tests passed ===")
