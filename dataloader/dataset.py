import csv
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Nutrition_RGBD_CSV(Dataset):
    """RGB-D 数据集（无食材输入）。"""

    def __init__(self, imagery_root, split_csv_path, transform=None):
        self.imagery_root = imagery_root
        self.transform = transform

        self.dish_ids = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []

        with open(split_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dish_id = row['dish_id']
                self.dish_ids.append(dish_id)
                self.total_calories.append(np.array(float(row['calories']), dtype=np.float32))
                self.total_mass.append(np.array(float(row['mass']), dtype=np.float32))
                self.total_fat.append(np.array(float(row['fat']), dtype=np.float32))
                self.total_carb.append(np.array(float(row['carb']), dtype=np.float32))
                self.total_protein.append(np.array(float(row['protein']), dtype=np.float32))

    def __getitem__(self, index):
        dish_id = self.dish_ids[index]
        dish_dir = os.path.join(self.imagery_root, 'realsense_overhead', dish_id)
        rgb_path = os.path.join(dish_dir, 'rgb.png')
        depth_path = os.path.join(dish_dir, 'depth_color.png')

        img_rgb = cv2.imread(rgb_path)
        img_depth = cv2.imread(depth_path)

        if img_rgb is None or img_depth is None:
            raise FileNotFoundError(
                f"Image missing in {dish_dir}: rgb={img_rgb is None}, depth={img_depth is None}")

        img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        img_depth = Image.fromarray(cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_depth = self.transform(img_depth)

        targets = torch.tensor([
            self.total_calories[index],
            self.total_mass[index],
            self.total_fat[index],
            self.total_carb[index],
            self.total_protein[index],
        ], dtype=torch.float32)

        return img_rgb, img_depth, targets, dish_id

    def __len__(self):
        return len(self.dish_ids)


_INDEX_CACHE = {}


def _load_index_json(index_path):
    """加载预处理索引 JSON，并在模块级缓存复用。"""
    cache_key = (os.path.abspath(index_path), os.path.getmtime(index_path))
    if cache_key not in _INDEX_CACHE:
        with open(index_path, 'r', encoding='utf-8') as f:
            _INDEX_CACHE[cache_key] = json.load(f)
    return _INDEX_CACHE[cache_key]


class Nutrition_RGBD_Ingredient(Dataset):
    """RGB + Depth + 食材列表 → 5 项营养标签。

    优先读取预处理索引 dish_ingredient_index.json，索引缺失时回退解析
    dish_ingredients.csv。

    返回 (6 元组):
        img_rgb [3,H,W], img_depth [3,H,W], ingr_ids [N],
        mask [N], targets [5], dish_id
    return_grams=True 时返回 7 元组（grams 插在 ingr_ids 与 mask 之间）。
    """

    def __init__(self, data_root, split_csv_path, vocab_path,
                 transform=None, max_ingredients=20,
                 index_path=None, return_grams=False):
        self.data_root = data_root
        self.imagery_root = os.path.join(data_root, 'imagery')
        self.transform = transform
        self.max_ingredients = max_ingredients
        self.return_grams = return_grams

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.num_ingredients = self.vocab['num_ingredients']
        self.ingr_id_to_idx = self.vocab['ingr_id_to_idx']

        # 防止标签泄露：vocab 不应包含营养密度表
        if 'nutrition_table' in self.vocab:
            raise RuntimeError(
                "检测到 vocab 中仍包含 nutrition_table 字段。请重新运行 "
                "scripts/build_dataset_index.py 重建索引，该字段构成对 "
                "mass/cal 标签的确定性泄露。"
            )

        print(f'Nutrition_RGBD_Ingredient: num_ingredients={self.num_ingredients}, '
              f'max_ingredients={max_ingredients}')

        self.index_path = index_path or \
            os.path.join(self.imagery_root, 'dish_ingredient_index.json')
        self.dish_ingredients, self.dish_grams = self._load_dish_index()

        self.samples = []
        missing_in_index = 0
        with open(split_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dish_id = row['dish_id']
                if dish_id not in self.dish_ingredients:
                    missing_in_index += 1
                    continue
                self.samples.append({
                    'dish_id': dish_id,
                    'calories': float(row['calories']),
                    'mass': float(row['mass']),
                    'fat': float(row['fat']),
                    'carb': float(row['carb']),
                    'protein': float(row['protein']),
                })

        if missing_in_index:
            print(f'Warning: {missing_in_index} split dishes missing in ingredient index '
                  f'(skipped)')

        print(f'Loaded {len(self.samples)} samples with ingredients from {split_csv_path}')

    def _load_dish_index(self):
        """默认读预处理索引；需要 grams 或索引缺失时回退解析 dish_ingredients.csv。"""
        if self.return_grams:
            print('return_grams=True: using dish_ingredients.csv for real gram weights')
            return self._load_dish_ingredients_from_csv()

        if os.path.isfile(self.index_path):
            payload = _load_index_json(self.index_path)
            entries = payload.get('dish_id_to_ingredients', payload)
            dish_ingredients = {
                dish_id: entry.get('vocab_idx', [])
                for dish_id, entry in entries.items()
            }
            print(f'Loaded preprocessed dish index: {len(dish_ingredients)} dishes '
                  f'from {self.index_path}')
            return dish_ingredients, {}

        print(f'Warning: preprocessed index not found: {self.index_path}\n'
              f'         Falling back to dish_ingredients.csv parsing. Run '
              f'"python scripts/build_dataset_index.py" to speed up loading.')
        return self._load_dish_ingredients_from_csv()

    def _load_dish_ingredients_from_csv(self):
        csv_path = os.path.join(self.data_root, 'dish_ingredients.csv')
        dish_ingredients, dish_grams = {}, {}
        seen = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                did = row['dish_id']
                ingr_id = row['ingr_id']
                dish_seen = seen.setdefault(did, set())
                if ingr_id in dish_seen:
                    continue
                dish_seen.add(ingr_id)
                dish_ingredients.setdefault(did, []).append(
                    self.ingr_id_to_idx.get(ingr_id, 1))
                dish_grams.setdefault(did, []).append(float(row['grams']))
        return dish_ingredients, dish_grams

    def __getitem__(self, index):
        sample = self.samples[index]
        dish_id = sample['dish_id']
        dish_dir = os.path.join(self.imagery_root, 'realsense_overhead', dish_id)

        rgb_path = os.path.join(dish_dir, 'rgb.png')
        depth_path = os.path.join(dish_dir, 'depth_color.png')

        img_rgb = cv2.imread(rgb_path)
        img_depth = cv2.imread(depth_path)

        if img_rgb is None or img_depth is None:
            raise FileNotFoundError(
                f"Image missing in {dish_dir}: rgb={img_rgb is None}, depth={img_depth is None}")

        img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        img_depth = Image.fromarray(cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_depth = self.transform(img_depth)

        idx_list = self.dish_ingredients.get(dish_id, [])
        grams_list = self.dish_grams.get(dish_id, [])
        N = min(len(idx_list), self.max_ingredients)
        ingr_ids = torch.zeros(self.max_ingredients, dtype=torch.long)
        mask = torch.zeros(self.max_ingredients, dtype=torch.bool)

        for i in range(N):
            ingr_ids[i] = min(max(idx_list[i], 0), self.num_ingredients - 1)
            mask[i] = True

        targets = torch.tensor([
            sample['calories'], sample['mass'],
            sample['fat'], sample['carb'], sample['protein'],
        ], dtype=torch.float32)

        if self.return_grams:
            grams = torch.zeros(self.max_ingredients, dtype=torch.float32)
            for i in range(min(N, len(grams_list))):
                grams[i] = grams_list[i]
            return img_rgb, img_depth, ingr_ids, grams, mask, targets, dish_id

        return img_rgb, img_depth, ingr_ids, mask, targets, dish_id

    def __len__(self):
        return len(self.samples)
