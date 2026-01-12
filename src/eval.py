import os
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import hamming_loss, jaccard_score
from sklearn.metrics import multilabel_confusion_matrix
import torch.nn as nn
import torch
import wandb

from helper_functions import load_mixed_datasets
from torch.utils.data import DataLoader
from naive_baseline import (
    naive_baseline_all_zeros,
    naive_baseline_all_ones,
    naive_baseline_random,
    naive_baseline_fixed_mix_size
)

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    mix_size,
    threshold=0.5
):
    model.eval()

    all_targets = []
    all_preds = []
    running_loss = 0.0
    num_batches = 0

    for images, targets in tqdm.tqdm(dataloader, desc="Testing"):
        images = images.to(device)
        #targets = one_hot_encode(targets, num_classes=10)
        targets = targets.to(device).float()

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()

        all_targets.append(targets.cpu())
        all_preds.append(preds.cpu())

        running_loss += loss.item()
        num_batches += 1

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    res = compute_metrics_from_predictions(all_targets, all_preds, mix_size)
    res['eval_loss'] = running_loss / num_batches

    return res


def compute_metrics_from_predictions(all_targets, all_preds, mix_size):
    """
    Compute evaluation metrics from predictions and targets.
    
    Args:
        all_targets: numpy array of shape (n_samples, n_labels)
        all_preds: numpy array of shape (n_samples, n_labels)
        mix_size: int, number of classes mixed in each sample
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    accuracy = (all_targets * all_preds).sum(axis=1).mean()
    hamming = hamming_loss(all_targets, all_preds)
    jaccard = jaccard_score(all_targets, all_preds, average='samples')
    ml_confusion_mat = multilabel_confusion_matrix(all_targets, all_preds)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'accuracy': accuracy / mix_size,
        'hamming': hamming,
        'jaccard': jaccard,
        'multilabel_confusion_matrix': ml_confusion_mat
    }

def run_test(
        model,
        train_exp_name,
        mix_size,
        classes_list,
        classes_for_test,
        n_samples_per_mix,
        batch_size,
        train_dataset,
        test_dataset,
        random_seed):

    _ , test_mx = load_mixed_datasets(train_dataset,
                                      test_dataset,
                                      classes_list,
                                      classes_for_test,
                                      mix_size=mix_size,
                                      n_samples_per_mix=n_samples_per_mix,
                                      random_seed=random_seed)

    test_loader = DataLoader(test_mx, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss() 

    WANDB_ENTITY = os.environ.get('WANDB_ENTITY')
    WANDB_PROJECT = os.environ.get('WANDB_PROJECT')

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=WANDB_ENTITY,
        # Set the wandb project where this run will be logged.
        project=WANDB_PROJECT,
        name=f'test-{mix_size}-{n_samples_per_mix}',
        # Track hyperparameters and run metadata.
        config={            
            "architecture": str(model),
            "dataset": "CIFAR-10",
            "mix_size" : mix_size,
            "classes_list" : classes_list,
            "classes_for_test" : classes_for_test,
            "n_samples_per_mix" : n_samples_per_mix,
            "batch_size": batch_size,
            "random_seed" : random_seed,
            "n_test_samples" : len(test_mx),
            "mode" : 'test'
        }
    )

    val_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            mix_size=mix_size,
            threshold=0.5
        )
    
    val_metrics['experiment'] = 'test'

    run.log(val_metrics)    
    run.finish()
    
    # Naive baselines:
    num_classes = len(classes_list)
    
    # Baseline 1: All zeros
    baseline_zeros_metrics = naive_baseline_all_zeros(test_loader, mix_size, num_classes)
    baseline_zeros_metrics['experiment'] = 'nb_all_0'

    # Baseline 2: All ones
    baseline_ones_metrics = naive_baseline_all_ones(test_loader, mix_size, num_classes)
    baseline_ones_metrics['experiment'] = 'nb_all_1'

    # Baseline 3: Random (p=0.5)
    baseline_random_05_metrics = naive_baseline_random(test_loader, mix_size, num_classes, label_prob=0.5, random_seed=random_seed)
    baseline_random_05_metrics['experiment'] = 'nb_random_0.5'

    # Baseline 4: Random (p=mix_size/num_classes)
    prob = mix_size / num_classes
    baseline_random_prop_metrics = naive_baseline_random(test_loader, mix_size, num_classes, label_prob=prob, random_seed=random_seed)
    baseline_random_prop_metrics['experiment'] = 'nb_random_p'

    # Baseline 5: Fixed mix size (exactly mix_size labels per sample)
    baseline_fixed_mix_metrics = naive_baseline_fixed_mix_size(test_loader, mix_size, num_classes, random_seed=random_seed)
    baseline_fixed_mix_metrics['experiment'] = 'nb_random_fixsize'


    for eval_met in [
        val_metrics,
        baseline_zeros_metrics,
        baseline_ones_metrics,
        baseline_random_05_metrics,
        baseline_random_prop_metrics,
        baseline_fixed_mix_metrics
    ]:
    
        nb_exp = eval_met['experiment']
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=WANDB_ENTITY,
            # Set the wandb project where this run will be logged.
            project=WANDB_PROJECT,
            name=f'{nb_exp}-{mix_size}-{n_samples_per_mix}-{train_exp_name}',
            # Track hyperparameters and run metadata.
            config={            
                "architecture": type(model).__name__,
                "dataset": "CIFAR-10",
                "mix_size" : mix_size,
                "classes_list" : classes_list,
                "classes_for_test" : classes_for_test,
                "n_samples_per_mix" : n_samples_per_mix,
                "batch_size": batch_size,
                "random_seed" : random_seed,
                "n_test_samples" : len(test_mx),
                "mode" : 'test'
            }
        )
        run.log(eval_met)
        run.finish()
        eval_loss = eval_met.get('eval_loss', -1)
        print(
                f"Experiment: {eval_met['experiment']}"
                f"Val Loss: {eval_loss:.4f} | "
                f"Micro F1: {eval_met['micro_f1']:.3f} | "
                f"Macro F1: {eval_met['macro_f1']:.3f} | "
                f"Accuracy: {eval_met['accuracy']:.3f}\n"
                f"Jaccard: {eval_met['jaccard']} | "
                f"Hamming: {eval_met['hamming']} | "
                f"Multi-Label Confusion Matrix:\n"
                f"{eval_met['multilabel_confusion_matrix']}"
        )


    return val_metrics

