# QuatODE-Net

基于四元数 Neural ODE 的多模态营养估计网络。输入俯拍食物 RGB 图像、深度图像和食材列表，回归 `calories`、`mass`、`fat`、`carb`、`protein` 五项营养指标。

## 方法概览

```text
RGB   ──> MambaVision-T ──> 三级视觉特征 ──┐
Depth ──> MambaVision-T ──> 三级视觉特征 ──┼──> 四元数 ODE 融合 ──> 标签路由 ──> 5 项营养指标
Ingr  ──> Text Encoder ───────────────────┘
```

核心设计：

- **多尺度视觉编码**：RGB 与 Depth 分别使用独立的 MambaVision-T，取前 3 级特征（160/320/640）。
- **食材文本编码**：食材 ID 列表经 Transformer 编码为类别语义向量，不使用 grams，避免份量信息泄露。
- **四元数状态**：每个尺度构造 `q = (w, r, d, t)`，其中 RGB、Depth、Text 分别映射到三个虚轴。
- **Hamilton 力场**：使用四元数线性层与 Hamilton 乘积构造 ODE 力场 `f(q)`。
- **辛积分**：采用 Leapfrog 格式演化 `num_ode_steps` 步，动量初始化为零，无新增参数。
- **轨迹聚合**：对包含初始状态在内的全程模长 `||q||` 等权平均，得到每个尺度的共识向量。
- **后置标签路由**：5 个标签查询对三尺度共识与三模态 token 做交叉注意力，以零初始化残差方式精炼预测。
- **损失函数**：`NormalizedMAELoss`，按各营养指标的平均绝对值归一化后求和。

## 目录结构

```text
QuatODE-Net/
├── dataloader/
│   ├── dataloader.py          # DataLoader 与 QuatODE-Net 专用 loader
│   └── dataset.py             # Nutrition5K RGB-D + 食材数据集
├── model/
│   ├── mambavision_backbone.py
│   ├── text_encoder.py
│   ├── quat_ode_fusion.py     # 四元数 ODE 融合、辛积分、轨迹聚合
│   ├── label_router.py        # PostFusionLabelRouter
│   ├── quat_ode_net.py        # 完整网络
│   └── loss.py
├── process_data/
│   └── split_dataset.py       # 生成 train.csv / test.csv
├── scripts/
│   ├── build_dataset_index.py # 构建词表与食材索引
│   ├── train_quat_ode.py
│   └── evaluate_quat_ode.py
├── weights/                   # 预训练骨干权重
├── requirements.txt
└── README.md
```

## 环境配置

推荐 Python 3.10+ 与 CUDA 环境。

```bash
conda create -n quatode python=3.10 -y
conda activate quatode

# 根据本机 CUDA 版本选择 PyTorch 安装命令
pip install torch torchvision
pip install -r requirements.txt
pip install einops
```

`requirements.txt` 包含 OpenCV、NumPy 与 tqdm；`einops` 为 MambaVision 骨干所需。若已安装 `mamba-ssm`，代码会自动使用其 CUDA 加速版 selective scan，否则回退到内置的纯 PyTorch 实现：

```bash
pip install mamba-ssm
```

## 数据准备

从 Nutrition5K 官方发布中获取数据，并按以下结构放置：

```text
dataset/nutrition5k_dataset/
├── dish_ingredients.csv
├── dish_nutrition_values.csv
├── ingredients_metadata.csv
└── imagery/
    └── realsense_overhead/
        └── <dish_id>/
            ├── rgb.png
            └── depth_color.png
```

生成训练/测试划分：

```bash
python process_data/split_dataset.py \
  --data_root dataset/nutrition5k_dataset \
  --train_ratio 0.8333 \
  --seed 42
```

构建食材词表与菜品-食材索引：

```bash
python scripts/build_dataset_index.py \
  --data_root dataset/nutrition5k_dataset \
  --require_image
```

该步骤会生成：

- `imagery/ingredient_vocab.json`：食材 ID 到词表索引的映射；
- `imagery/dish_ingredient_index.json`：每个菜品的食材索引，供训练时快速加载。

索引只保存食材类别信息，不包含 grams 或营养密度表，以避免对 `mass` / `calories` 标签的确定性泄露。

## 预训练骨干权重

将 MambaVision-T 的 ImageNet-1K 权重放到：

```text
weights/mambavision_tiny_1k.pth.tar
```

权重可从 [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision) 获取。若文件不存在，训练脚本会打印警告并使用随机初始化骨干。

## 训练

```bash
python scripts/train_quat_ode.py \
  --data_root dataset/nutrition5k_dataset \
  --vocab_path dataset/nutrition5k_dataset/imagery/ingredient_vocab.json \
  --backbone_weights weights/mambavision_tiny_1k.pth.tar \
  --epochs 150 \
  --b 32 \
  --output_dir runs \
  --exp_tag final
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data_root` | `dataset` | Nutrition5K 根目录或 `nutrition5k_dataset` 目录 |
| `--vocab_path` | `dataset/nutrition5k_dataset/imagery/ingredient_vocab.json` | 食材词表路径 |
| `--backbone_weights` | `weights/mambavision_tiny_1k.pth.tar` | 预训练骨干权重 |
| `--epochs` | `150` | 最大训练轮数 |
| `--b` | `32` | batch size |
| `--lr` | `1e-4` | 非骨干参数学习率 |
| `--lr_backbone` | `1e-5` | 骨干学习率 |
| `--weight_decay` | `0.05` | AdamW 权重衰减 |
| `--num_ode_steps` | `4` | ODE 演化步数 |
| `--latent_dim` | `384` | 融合潜空间维度 |
| `--head_dropout` | `0.3` | 预测头 dropout |
| `--output_dir` | `runs` | 输出根目录 |
| `--exp_tag` | `final` | 实验标签，用于命名输出目录 |

训练使用 AdamW、余弦退火学习率与早停（默认 patience 为 30）。固定随机种子为 42。

## 评估

```bash
python scripts/evaluate_quat_ode.py \
  --weights runs/final_<timestamp>/best_model.pth \
  --data_root dataset/nutrition5k_dataset \
  --vocab_path dataset/nutrition5k_dataset/imagery/ingredient_vocab.json
```

`--weights` 也可以传入包含 `best_model.pth` 的运行目录。评估结果会保存到权重同目录下的 `eval_quat_ode_result.json`。

如需修改过 `--text_dim`、`--latent_dim`、`--num_ode_steps`、`--head_hidden` 或 `--head_dropout`，评估时必须传入与训练相同的值（默认值与训练脚本一致）。

## 输出文件

每次训练会生成：

```text
runs/<exp_tag>_<timestamp>/
├── config.json            # 模型与训练配置
├── train_log.jsonl        # 逐 epoch 训练/测试指标
├── best_model.pth         # 测试 MAE 最优的模型
├── final_summary.json     # 最终逐指标结果
└── attn_analysis.pt       # 标签-键注意力归因
```

`attn_analysis.pt` 包含 5 个营养标签对 6 个键的注意力均值与熵：

```text
键顺序：S1, S2, S3, RGB, Depth, Text
```

## 复现说明

- 数据集与预训练权重不包含在本仓库中，需要自行下载。
- 训练默认使用 MambaVision-T；`model/__init__.py` 同时暴露了 MambaVision-S/B 构造器，便于扩展。
- 深度图按 RGB 三通道方式读取并与 RGB 使用相同的归一化流程。
- 若显存不足，可减小 `--b`；若训练数据加载较慢，可增大 `--num_workers`。