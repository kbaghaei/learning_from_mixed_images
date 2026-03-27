# Learning from Mixed Images

This project investigates whether a Vision Transformer (ViT) can learn to identify individual classes from images that are formed by **pixel-wise addition of multiple source images** from different classes. The core question: if a model is trained only on these mixed images, can it recover the constituent class labels?

## Problem Setup

Each training sample is a **mixed image** created by adding together `mix_size` images from distinct CIFAR-10 classes. The model must predict a **multi-hot label vector** indicating which classes were mixed into the image. This is a multi-label classification problem.

A key experimental design choice: the combinations of classes used at test time are **held out during training** (controlled via `classes_for_test`). This tests whether the model genuinely learns class-level representations, or merely memorizes co-occurrence patterns.

## Project Structure

```
learning_from_mixed_images/
├── src/
│   ├── run.py                # Entry point — loads configs and orchestrates train + eval
│   ├── train.py              # Training loop with warmup-cosine LR schedule
│   ├── eval.py               # Evaluation metrics and test runner
│   ├── models.py             # ViT model definitions
│   ├── mixer_dataset.py      # Dataset that creates and serves mixed images
│   ├── helper_functions.py   # Data loading, transforms, combination generation
│   ├── naive_baseline.py     # Naive baselines for comparison
│   └── early_stopper.py      # Early stopping utility
├── experiments/
│   ├── configs.json          # Experiment configurations
│   └── configs_debug.json    # Debug configurations (smaller scale)
└── wandb_config.json         # W&B credentials (not tracked)
```

## Models

All models are defined in [src/models.py](src/models.py).

| Model | Patch Embedding | Notes |
|---|---|---|
| `MiniViT` | Conv2d | Single-label baseline ViT |
| `MultiLabelMiniConViT` | Conv2d | Multi-label head |
| `MultiLabelMiniViT` | Linear (no conv) | Multi-label head, used in experiments |
| `MultiLabelWideMiniViT` | Linear, wide images | For concatenated-image inputs |

The default training model is `MultiLabelMiniViT`: a small ViT with 4 transformer blocks, embed dim 128, 4 attention heads, patch size 4, operating on 32×32 CIFAR-10 images.

## Dataset: `MixerDataset`

[src/mixer_dataset.py](src/mixer_dataset.py) generates mixed samples on the fly at dataset creation time:

1. All combinations of `mix_size` classes are enumerated from `classes_list`.
2. Combinations that include only classes from `classes_for_test` are held out for testing.
3. For each training combination, `n_samples_per_mix` mixed images are created by pixel-wise addition of one image sampled from each constituent class.
4. Labels are multi-hot encoded vectors of length `num_classes`.

`SampleIndexDealer` ensures balanced sampling across classes (shuffle-without-replacement, cycling).

## Experiments

Experiments are defined in [experiments/configs.json](experiments/configs.json). Each experiment trains on one `mix_size` and tests across multiple mix sizes to probe generalization.

| Experiment | Train mix_size | Test mix_sizes |
|---|---|---|
| `experiment_1` | 1 (single images) | 1 |
| `experiment_2` | 2 | 1, 2, 3 |
| `experiment_3` | 3 | 1, 2, 3 |

**Training:** Classes 0–9 (all CIFAR-10), with combinations involving only classes 0–4 held out for testing.

## Training Details

- **Loss:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.05)
- **Schedule:** Linear warmup (10 epochs) → cosine decay
- **Augmentation:** RandomCrop, RandomHorizontalFlip, RandAugment, RandomErasing

## Evaluation

Metrics reported per run (tracked with Weights & Biases):

- Micro / Macro F1
- Accuracy (fraction of correct labels per sample, normalized by mix_size)
- Hamming loss
- Jaccard score (samples average)
- Multi-label confusion matrix

Each test run also compares against five **naive baselines**:

| Baseline | Description |
|---|---|
| All zeros | Predict no labels |
| All ones | Predict all labels |
| Random (p=0.5) | Flip each label independently with p=0.5 |
| Random (p=mix_size/C) | Calibrated random: expected number of positives = mix_size |
| Fixed mix size | Randomly pick exactly `mix_size` labels per sample |

## Setup & Usage

**Requirements:** PyTorch, torchvision, scikit-learn, wandb, fire, tqdm, Pillow, numpy

**Configure W&B** by creating `wandb_config.json` in the project root:
```json
{
    "entity": "your-wandb-entity",
    "project": "your-project-name",
    "wandb_api_key": "your-api-key"
}
```

**Run an experiment:**
```bash
cd src
python run.py --experiment_name experiment_2
```

**Debug mode** (uses `configs_debug.json` for smaller-scale runs):
```bash
python run.py --experiment_name experiment_2 --debug True
```