import torch
import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR
import json
import os
from eval import evaluate


def warmup_cosine_scheduler(
    optimizer,
    warmup_epochs,
    total_epochs,
    min_lr_ratio=0.1,
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    scaler=None,   # for mixed precision
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, targets in tqdm.tqdm(dataloader, desc="Training"):
        images = images.to(device)
        #targets = one_hot_encode(targets, num_classes=10)
        targets = targets.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches





from helper_functions import load_mixed_datasets
from torch.utils.data import DataLoader
from models import MultiLabelMiniConViT, MultiLabelMiniViT
import wandb
import torch.nn as nn

def run_train(
    mix_size,
    classes_list,
    classes_for_test,
    n_samples_per_mix,
    batch_size,
    train_dataset,
    test_dataset,
    total_epochs,
    lr=3e-4,
    weight_decay=0.05,
    warmup_epochs=5,
    random_seed=0):

    train_mx, test_mx = load_mixed_datasets(train_dataset, test_dataset, classes_list, classes_for_test, mix_size=mix_size, n_samples_per_mix=n_samples_per_mix, random_seed=random_seed)

    print(f'Loaded train dataset: {len(train_mx)}')
    print(f'Loaded test dataset: {len(test_mx)}')

    train_loader = DataLoader(train_mx, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_mx, batch_size=batch_size, shuffle=False)


    model = MultiLabelMiniViT()
    #MultiLabelMiniConViT()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs
    )

    criterion = nn.BCEWithLogitsLoss()  # multi-label

    WANDB_ENTITY = os.environ.get('WANDB_ENTITY')
    WANDB_PROJECT = os.environ.get('WANDB_PROJECT')

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=WANDB_ENTITY,
        # Set the wandb project where this run will be logged.
        project=WANDB_PROJECT,
        name=f'train-{mix_size}-{n_samples_per_mix}',
        # Track hyperparameters and run metadata.
        config={            
            "architecture": str(model),
            "dataset": "CIFAR-10",
            "mix_size" : mix_size,
            "classes_list" : classes_list,
            "classes_for_test" : classes_for_test,
            "n_samples_per_mix" : n_samples_per_mix,
            "batch_size": batch_size,
            "total_epochs" : total_epochs,
            "lr" :lr,
            "weight_decay" : weight_decay,
            "warmup_epochs" : warmup_epochs,
            "random_seed" : random_seed,
            "n_train_samples" : len(train_mx),
            "n_test_samples" : len(test_mx),
            "mode" : 'train'
        }
    )

    # Train the model.
    best_model_state = None
    best_val_loss = float('inf')
    best_model_epoch = 0

    for epoch in range(total_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            scaler=None,
        )

        val_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            mix_size=mix_size,
            threshold=0.5
        )

        # store in wandb
        if val_metrics['eval_loss'] < best_val_loss:
            best_val_loss = val_metrics['eval_loss']
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_model_epoch = epoch

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['eval_loss']:.4f} | "
            f"Micro F1: {val_metrics['micro_f1']:.3f} | "
            f"Macro F1: {val_metrics['macro_f1']:.3f} | "
            f"Accuracy: {val_metrics['accuracy']:.3f}\n"
            f"Jaccard: {val_metrics['jaccard']} | "
            f"Hamming: {val_metrics['hamming']} | "
            f"Multi-Label Confusion Matrix:\n"
            f"{val_metrics['multilabel_confusion_matrix']}"
        )
        val_metrics['train_loss'] = train_loss
        val_metrics['best_model_epoch'] = best_model_epoch
        print('best model at epoch: ', best_model_epoch)
        run.log(val_metrics)
    
    run.finish()

    model.load_state_dict(best_model_state)
    model.to(device)

    return model
