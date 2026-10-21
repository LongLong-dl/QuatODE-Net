"""构建训练用的 ingredient_vocab.json 与 dish_ingredient_index.json。

索引仅保存食材类别语义，不写入 grams / 营养密度，避免标签泄露。
"""

import argparse
import csv
import json
import os
from collections import OrderedDict, Counter


_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATA_ROOT = os.path.join(_PROJ_ROOT, 'dataset', 'nutrition5k_dataset')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build ingredient vocab + dish_id -> ingredient index '
                    '(preprocessing for fast training-time sample loading)')
    parser.add_argument('--data_root', type=str, default=_DEFAULT_DATA_ROOT,
                        help='path to nutrition5k_dataset root')
    parser.add_argument('--min_freq', type=int, default=1,
                        help='Minimum occurrence count to keep ingredient. '
                             'Default=1 (no filtering) because Nutrition5k has only ~4K '
                             'training samples; filtering rare ingredients would collapse '
                             "many dishes' key ingredients into <unk>, erasing their "
                             'density priors.')
    parser.add_argument('--max_ingredients', type=int, default=20,
                        help='truncate each dish ingredient list to this length '
                             '(default: 20, matches training max_ingredients)')
    parser.add_argument('--no_dedup', action='store_true',
                        help='keep repeated ingredients within a dish (default: dedup)')
    parser.add_argument('--splits', type=str, nargs='*',
                        default=['train.csv', 'test.csv'],
                        help='split CSV filenames under <data_root>/imagery used to '
                             'annotate train/test (default: train.csv test.csv)')
    parser.add_argument('--restrict_to_splits', action='store_true',
                        help='only keep dishes that appear in the given splits')
    parser.add_argument('--require_image', action='store_true',
                        help='only keep dishes that have rgb.png and depth_color.png')
    parser.add_argument('--vocab_output', type=str, default=None,
                        help='output vocab JSON path '
                             '(default: <data_root>/imagery/ingredient_vocab.json)')
    parser.add_argument('--index_output', type=str, default=None,
                        help='output dish index JSON path '
                             '(default: <data_root>/imagery/dish_ingredient_index.json)')
    return parser.parse_args()


def load_ingredient_names(csv_path):
    """ingredients_metadata.csv -> {ingr_id: ingr_name}（不解析营养密度）"""
    names = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[row['ingr_id']] = row['ingr_name'].strip()
    return names


def count_ingredient_occurrences(dish_csv_path):
    counter = Counter()
    with open(dish_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counter[row['ingr_id']] += 1
    return counter


def load_dish_ingredients(csv_path, dedup=True):
    """dish_ingredients.csv -> OrderedDict{dish_id: [(ingr_id, csv_name), ...]}

    顺序按 CSV 首次出现；dedup=True 时同菜重复成分只保留一次。
    """
    dishes = OrderedDict()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_id = row['dish_id']
            ingr_id = row['ingr_id']
            name = (row.get('ingr_name') or '').strip()
            entry = dishes.setdefault(dish_id, [])
            if dedup and any(iid == ingr_id for iid, _ in entry):
                continue
            entry.append((ingr_id, name))
    return dishes


def build_vocab(id_to_name, freq, min_freq):
    """构建 ingr_id ↔ idx 词表（0=<pad>, 1=<unk>），返回 (vocab, kept)"""
    kept = [ingr_id for ingr_id in sorted(id_to_name.keys())
            if freq.get(ingr_id, 0) >= min_freq]

    ingr_id_to_idx = {'<pad>': 0, '<unk>': 1}
    idx_to_name = {0: '<pad>', 1: '<unk>'}
    idx_to_ingr_id = {0: None, 1: None}
    for idx, ingr_id in enumerate(kept, start=2):
        ingr_id_to_idx[ingr_id] = idx
        idx_to_name[idx] = id_to_name[ingr_id]
        idx_to_ingr_id[idx] = ingr_id

    # 不写入 nutrition_table，避免 mass/cal 标签泄露
    vocab = {
        'ingr_id_to_idx': ingr_id_to_idx,
        'idx_to_name': {str(k): v for k, v in idx_to_name.items()},
        'idx_to_ingr_id': {str(k): v for k, v in idx_to_ingr_id.items()},
        'num_ingredients': len(ingr_id_to_idx),
    }
    return vocab, kept


def load_splits(imagery_dir, split_files):
    """-> {dish_id: split_name}"""
    dish_split = {}
    for name in split_files:
        path = os.path.join(imagery_dir, name)
        if not os.path.isfile(path):
            print(f'Warning: split file not found, skipped: {path}')
            continue
        tag = os.path.splitext(name)[0]
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dish_split[row['dish_id']] = tag
    return dish_split


def has_image(imagery_dir, dish_id):
    d = os.path.join(imagery_dir, 'realsense_overhead', dish_id)
    return os.path.isfile(os.path.join(d, 'rgb.png')) and \
        os.path.isfile(os.path.join(d, 'depth_color.png'))


def main():
    args = parse_args()

    dish_csv = os.path.join(args.data_root, 'dish_ingredients.csv')
    meta_csv = os.path.join(args.data_root, 'ingredients_metadata.csv')
    imagery_dir = os.path.join(args.data_root, 'imagery')
    vocab_output = args.vocab_output or \
        os.path.join(imagery_dir, 'ingredient_vocab.json')
    index_output = args.index_output or \
        os.path.join(imagery_dir, 'dish_ingredient_index.json')

    for path in (dish_csv, meta_csv):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Required file not found: {path}')

    id_to_name = load_ingredient_names(meta_csv)
    freq = count_ingredient_occurrences(dish_csv)
    vocab, kept = build_vocab(id_to_name, freq, args.min_freq)
    print(f'Loaded {len(id_to_name)} ingredient names from metadata')
    print(f'Vocabulary: num_ingredients={vocab["num_ingredients"]} '
          f'(kept={len(kept)}, min_freq={args.min_freq})')

    os.makedirs(os.path.dirname(vocab_output), exist_ok=True)
    with open(vocab_output, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f'Saved vocab ({vocab["num_ingredients"]} entries) to {vocab_output}')

    filtered = [iid for iid in id_to_name if freq.get(iid, 0) < args.min_freq]
    kept_freqs = sorted(freq.get(iid, 0) for iid in kept)
    if kept_freqs:
        mid = len(kept_freqs) // 2
        median = kept_freqs[mid] if len(kept_freqs) % 2 == 1 else \
            (kept_freqs[mid - 1] + kept_freqs[mid]) / 2
        print(f'Frequency distribution (kept): min={kept_freqs[0]}, '
              f'max={kept_freqs[-1]}, median={median}')
    print(f'Filtered out: {len(filtered)} (min_freq={args.min_freq})')
    print(f'Vocab fields: {list(vocab.keys())}')

    dishes = load_dish_ingredients(dish_csv, dedup=not args.no_dedup)
    dish_split = load_splits(imagery_dir, args.splits)
    id_to_idx = vocab['ingr_id_to_idx']

    total_occurrences = sum(len(v) for v in dishes.values())
    truncated = 0
    missing_names = Counter()
    name_mismatch = Counter()
    dropped_no_split = 0
    dropped_no_image = 0

    result = OrderedDict()
    for dish_id, items in dishes.items():
        if args.restrict_to_splits and dish_id not in dish_split:
            dropped_no_split += 1
            continue
        if args.require_image and not has_image(imagery_dir, dish_id):
            dropped_no_image += 1
            continue

        if len(items) > args.max_ingredients:
            truncated += 1
            items = items[:args.max_ingredients]

        ingr_ids, ingr_names, vocab_idx = [], [], []
        for ingr_id, csv_name in items:
            meta_name = id_to_name.get(ingr_id)
            if meta_name is None:
                missing_names[ingr_id] += 1
                name = csv_name or ingr_id
            else:
                name = meta_name
                if csv_name and csv_name != meta_name:
                    name_mismatch[(csv_name, meta_name)] += 1
            ingr_ids.append(ingr_id)
            ingr_names.append(name)
            vocab_idx.append(int(id_to_idx.get(ingr_id, 1)))

        result[dish_id] = {
            'ingr_ids': ingr_ids,
            'ingr_names': ingr_names,
            'vocab_idx': vocab_idx,
            'n_ingredients': len(ingr_ids),
            'split': dish_split.get(dish_id, None),
        }

    counts = [v['n_ingredients'] for v in result.values()]
    meta = {
        'source_dish_ingredients': dish_csv,
        'source_metadata': meta_csv,
        'vocab_path': vocab_output,
        'num_ingredients': vocab['num_ingredients'],
        'max_ingredients': args.max_ingredients,
        'dedup': not args.no_dedup,
        'num_dishes': len(result),
        'fields': ['ingr_ids', 'ingr_names', 'vocab_idx', 'n_ingredients', 'split'],
        'note': 'No grams / no nutrition density: text modality only carries '
                'ingredient category semantics to avoid label leakage.',
    }
    payload = {'meta': meta, 'dish_id_to_ingredients': result}

    os.makedirs(os.path.dirname(index_output), exist_ok=True)
    with open(index_output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'\nSaved {len(result)} dishes to {index_output}')

    print('\n=== Diagnostics ===')
    print(f'Dish-ingredient occurrences (raw): {total_occurrences}')
    print(f'Dishes kept: {len(result)}')
    if dropped_no_split:
        print(f'Dropped (not in splits): {dropped_no_split}')
    if dropped_no_image:
        print(f'Dropped (missing image): {dropped_no_image}')
    if counts:
        print(f'Ingredients per dish: min={min(counts)}, max={max(counts)}, '
              f'avg={sum(counts) / len(counts):.2f}')
    print(f'Dishes truncated to max_ingredients={args.max_ingredients}: {truncated}')
    unique_ref = {iid for items in dishes.values() for iid, _ in items}
    print(f'Unique ingredient IDs referenced: {len(unique_ref)}')
    if missing_names:
        print(f'Ingredient IDs missing in metadata: {len(missing_names)} '
              f'(e.g. {list(missing_names)[:5]})')
    if name_mismatch:
        print(f'Name mismatches (csv vs metadata): {len(name_mismatch)} '
              f'(e.g. {list(name_mismatch)[:3]})')
    split_counts = Counter(v['split'] for v in result.values())
    print(f'Split distribution: {dict(split_counts)}')


if __name__ == '__main__':
    main()
